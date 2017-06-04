#!/usr/bin/python
#
#  Utility class to parse the ImageMagick identify -verbose <image> command
#  and convert to various formats for further reuse by code or applications.
#
# Copyright (c) 2016, Metadata Technology North America Inc. (http://www.mtna.us)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of imagemagick-identify-parser nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import distutils.spawn
import os.path
import json
import lxml.etree as etree
import os
import re
from subprocess import PIPE, Popen
import sys
from xml.etree.ElementTree import ElementTree,SubElement,Element,dump,tostring


__doc__ = """
This module provides a indentation-based parser that parses
the metadata provided by ImageMagick's utility called 'identify'.

The code can be used both as a module or as a command.
"""

def checkProgram(program):
    """
    Check if a program is available
    """
    return distutils.spawn.find_executable(program)


class ImageMagickIdentifyParser:
    """
    DICOM image metadata indentation-based parser class
    """

    optHistogram = False
    Data = None
    HISTOGRAM_ELEM="HistogramLevel"
    # RE_GROUPED_ENTRY examples:
    # dcm:DeviceSerialNumber
    RE_GROUPED_ENTRY = r"(?P<prefix>.+?):(?P<name>.*)$"
    # RE_LINE_GENERIC examples:
    # Page geometry: 512x512+0+0
    RE_LINE_GENERIC = r"^(?P<leading>\s*)(?P<name>.*):(?P<value>\s.*|)$"
    # RE_LINE_HISTO examples:
    # 30489: (  385,  385,  385) #018101810181 gray(0.587472%,0.587472%,0.587472%)
    # 6709: (    0,    0,    0) #000000000000 gray(0,0,0)
    # 6709: (    0,    0,    0) #000000000000 gray(0)
    # 16680: (  128,  128,  128) #008000800080 gray(0.195315%)
    # 25206: (  256,  256,  256) #010001000100 gray(0.390631%)
    # 12: ( 8224,17219,23644,30583) #202043435C5C7777 srgba(32,67,92,0.466667)
    # 5672: (    0,    0,    0,65535) #000000000000 black
    #
    # Note that the last list of numbers in the parenthesis can have either 1,3 or 4 elements.
    # The regex part for colors and percentages is essentially the same, the only difference is that
    # percentages can also contain floating point and percentage signs.
    RE_LINE_HISTO = \
            r'\s*(?P<count>\d+):\s*'\
            r'\s*\('\
            r'(?P<colors>\s*\d+(?:,\s*\d+)*)+'\
            r'\s*\)\s*'\
            r'#(?P<hexval>[0-9A-F]{8,})\s*'\
            r'(?P<colorSpace>[a-zA-Z]+)\s*'\
            r'(?:'\
            r'\s*\('\
            r'(?P<percentages>\s*\d+(?:\.\d+%?)(?:,\s*\d+(?:\.\d+%?))*)'\
            r'\s*\)'\
            r')?'

    def __init__(self):
        if not checkProgram('identify'):
            raise Exception('[Error] ImageMagick is missing')

        # reset internal data
        self.Data = {}

    # adapt the tag name to conform with the XML format
    def normalizeName(self, name):
        """
        This method takes a string as parameter, converts it to camel-case
        and returns it
        """
        name = re.sub(r'\s','_',name)
        # we allow alphanumeric characters, and the colon
        # (because we'll split using that separator in treeTransformGroup)
        name = re.sub(r'[^a-zA-Z0-9:]','_',name)
        # trim any trailing underscores
        name = re.sub(r'_+$','',name)
        # trim any leading underscores
        name = re.sub(r'^_+','',name)

        def upperCallback(x):
            return x.group(1).upper()

        # apply regex to convert to camelCase
        # assuming it's in underscore format
        ccName = re.sub(r'_(.)',upperCallback,name)
        return ccName

    def runCmd(self, cmd):
        """
        This method runs a command and returns a list
        with the contents of its stdout and stderr and
        the exit code of the command.
        """
        p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
        (handleChildStdin,handleChildStdout,handleChildStderr) = (p.stdin, p.stdout, p.stderr)
        childStdout = handleChildStdout.read()
        childStderr = handleChildStderr.read()
        p.wait()
        return [childStdout, childStderr, p.returncode]

    def getIMVersion(self):
        stdout, stderr, exitcode = self.runCmd('identify -version')
        match = re.match(r'^.*ImageMagick (?P<version>\d+(?:\.\d+)+(?:-\d+)?)', stdout)

        if match and match.groupdict():
            d = match.groupdict()
            return d['version']

        return None

    def parseLineGeneric(self, line):
        """
        This method parses a generic line and returns it as a dict
        """
        matchGeneric = re.match(self.RE_LINE_GENERIC, line, re.UNICODE)
        if not matchGeneric:
            return None

        # extract the values picked up by the regex above
        d = matchGeneric.groupdict()
        name = d['name']
        value = d['value']
        leading = d['leading']

        # clean up leading and trailing whitespace
        value = re.sub(r'^\s+','',value)
        value = re.sub(r'\s+$','',value)

        # get the current level using the indentation
        lc = len(leading)/2
        # everything is shifted one level deeper, because we have a root node
        lc += 1

        # a new node is created to store the information extracted
        new_node = {
                'name': name, \
                'value': value, \
                'children': [], \
                'level': lc, \
                'parent': None,\
                }
        return new_node

    def parseLineHisto(self, line, level):
        """
        This method parses a histogram line and returns it as a dict
        """
        matchHisto = re.match(self.RE_LINE_HISTO, line, re.UNICODE)
        if not matchHisto:
            return None
        d = matchHisto.groupdict()

        # unroll colors named capture
        if 'colors' in d:
            colors = map(lambda x: int(x), d['colors'].split(','))
            d['colors'] = colors

        # unroll percentages named capture
        if 'percentages' in d:
            pStr = d['percentages']
            if pStr:
                pStr = re.sub(r'%','',pStr)
                p = pStr.split(',')
                percentages = map(lambda x: float(x), p)
                d['percentages'] = percentages

        newNode = d
        newNode['name'] = self.HISTOGRAM_ELEM
        newNode['value'] = ''
        newNode['children'] = []
        newNode['level'] = level
        newNode['parent'] = None
        return newNode

    def parse(self, filePath):
        """
        This method parses the metadata of an image file
        """
        if not os.path.isfile(filePath):
            raise Exception('[Error] The path does not point to a file')

        self.parseRaw(filePath)
        self.treeTransformGroup()

    def parseRaw(self, filePath):
        """
        This method takes as parameter a file path to an image, it then runs
        the identify command, retrieves the output and parses it into an abstract
        syntax tree.

        The identify commands' output is formatted as an space-indented
        tree, this method uses a stack for parsing. A node is built for
        each new line parsed and the node is placed on a stack.
        The node object is then assigned a parent(the previous node on
        the stack).

        After placing a new node on the stack, the stack will hold the entire
        path from the root to that node.

        Corner-case: identify also provides a color histogram with pixel counts
                     and colors ( http://www.imagemagick.org/Usage/files/#histogram )

                     The histogram lines have a varying indentation but don't
                     comply with the rest of the output. In order to
                     cover them, a custom regex was written that parses
                     those lines.
        """
        # get identify output
        output, error, exitcode = self.runCmd('identify -verbose ' + filePath)
        if exitcode != 0:
            raise Exception('[Error] Identify returned with non-zero exit code')

        output = output.decode('iso-8859-1').encode('utf-8')
        lines = output.split('\n')

        #### First pass: building the AST
        # initialize the stack with a root node
        stack = 200 * [None]
        root = {'children': [], 'parent': None, 'name': '', 'value': ''}
        stack[0] = root

        # flag that indicates the histogram parsing mode is on
        hm = False
        # initialize current level, previous level, histogram parsing level
        lc,lp,lh = 1,0,0

        for line in lines:
            newNode = None

            if hm:
                newNode = self.parseLineHisto(line, lh)
                # we failed parsing the histogram line, assume we're back to generic lines
                # the current line needs to be reparsed as a generic line
                if not newNode:
                    hm = False
                    newNode = self.parseLineGeneric(line)
            else:
                newNode = self.parseLineGeneric(line)
                if newNode and 'name' in newNode and newNode['name'] == 'Histogram':
                    # if we encounter the histogram node, we turn on histogram parsing mode
                    hm = True
                    # and we store the level for all the upcoming histogram lines that follow
                    lh = 1 + newNode['level']

            # skip all histogram lines if histogram parsing is off
            if hm and self.optHistogram == False:
                continue

            if newNode:
                lc = newNode['level']

                # dispose of the 'level' attribute, we only need that information here
                if 'level' in newNode:
                    lc = newNode['level']
                    del newNode['level']

                # set parent
                newNode['parent'] = stack[lc-1]
                # normalize name
                newNode['name'] = self.normalizeName(newNode['name'])
                # utf8 decode name and value
                newNode['name']  = newNode['name'].decode('utf-8')
                newNode['value'] = newNode['value'].decode('utf-8')
                # add the node as a child of its immediate parent
                (stack[lc-1])['children'].append(newNode)
                # put the node on the stack, this will subsequently be used
                # by the the next iterations of this loop(if this node has children)
                stack[lc] = newNode
                # update the previous level
                lp = lc

        # store the tree in the class attribute for later use
        self.Data = root

    def treeTransformGroup(self):
        """
        This method modifies the internal data in-place. It reorganizes the tree by
        grouping multiple nodes with the same prefix (the prefix is the string
        before the colon) into a new parent named after that common prefix.

        Example:

        <x>
          <p:a></p:a>
          <p:b></p:b>
        </x>

        =>

        <x>
          <p>
             <a></a>
             <b></b>
          </p>
        </x>
        """
        root = self.Data
        stack = []
        stack.append(root)

        while len(stack) > 0:
            # We're visiting the next immediate node on the stack
            # (regular DFS traversal)
            x = stack.pop()
            if 'children' in x:
                y = x['children']
                i = 0
                ## a's keys will be common prefixes
                ## and the values will be new parent nodes which hold the
                ## children that will be transfered from x
                a = {}
                while i < len(y):
                    z = y[i]
                    match = re.match(self.RE_GROUPED_ENTRY, z.get('name',''), re.UNICODE)
                    if match:
                        d = match.groupdict()
                        dPrefix = d['prefix']
                        dName = d['name']
                        if dName == '':
                            del y[i]
                            continue
                        p = a.setdefault(dPrefix, \
                                {
                                    'children': [],
                                    'name': dPrefix,
                                    'value': '',
                                    'parent': x,
                                })
                        # update z's parent because it has been moved.
                        # to illustrate this, here is how the hierarchy changes:
                        # x->z => x->p->z
                        # so p is the new parent of x
                        z['parent'] = p
                        z['name'] = dName
                        # add z to p's children
                        p['children'] += [z]
                        del y[i]
                    else:
                        i += 1
                # at this point, depending on whether there were children to be grouped
                # they were(into a), and those that couldn't be will have stayed the same(in x['children']).
                # some nodes (some children of x) have been displaced so they could be
                # attached to the newly created parent nodes.
                #
                # add newly created parent nodes to the tree
                x['children'] += a.values()

            # Get x's children and put them on the stack
            # (continue the regular DFS)
            if 'children' in x:
                stack += x['children']

    def treeTransformCompact(self, x):
        """
        This method rebuilds the tree in a more compact form.
        The method is specifically used in the preparation of the output
        for JSON format since the raw tree is too verbose.

        Summary of how this method works:
        1) we transform the tree as follows:
        we aim to replace the 'children' attribute with either
        an array or a dictionary, as follows:
        - all children have different names => we can store them in a dict
        - if at least two children have the same name => we need to store them in an array

        2) if a node has no children, and it has no additional attributes, then it
        can be expressed as {k: v}

        The return value of this method is always a list of 3 objects:
        - the type of node processed
        - the name of that node
        - the new node object

        The important part of the return value is the node object.
        """
        # check if it has no children
        xHasNoChildren = ('children' not in x) or ('children' in x and len(x['children']) == 0)
        # check if it has only basic properties: name,value,parent,children
        xHasOnlyNameValue = ('name' in x and 'value' in x and len(x.keys()) <= 4)

        # strip tree of parent attributes in order to avoid
        # circular references when serializing
        # to json
        del x['parent']

        if xHasNoChildren:
            del x['children']

            # x has no children
            if xHasOnlyNameValue:
                k = x['name']
                v = x['value']
                return [1,k,v]
            else:
                xname = x['name']
                xvalue = x['value']
                del x['name']
                # dispose of value attribute if it's empty
                if 'value' in x and x['value'] == '':
                    del x['value']
                return [2,xname,x]
        else:
            c = []
            xname = x['name']
            xvalue = x['value']
            # has children, recurse into children
            i = 0
            while i < len(x['children']):
                yi = x['children'][i]
                zi = self.treeTransformCompact(yi)
                c.append(zi)
                i += 1

            # after this point, the entire subtree rooted in x has been
            # rebuilt (except for x). the remainder of this method is
            # concerned with x and the way its children are represented.

            # constructing the new node
            w = None
            cnames = map(lambda z: z[1], c)
            if len(set(cnames)) == len(cnames):
                # the children all have distinct names, so w will be a dict
                w = {}
                for z in c:
                    w[z[1]] = z[2]
            else:
                # name collision are present, we need an array
                w = []
                for z in c:
                    w.append({z[1]: z[2]})

            return [3,xname,w]

    def serializeXML(self,root,xmlRoot):
        """
        Takes the internal root of the internal metadata tree
        and a root of an XML document as parameters.

        It populates the XML document with the metadata.
        """
        name = root['name']
        value = root['value']

        # serialize the node
        if 'children' in root and len(root['children']) > 0:
            for c in root['children']:
                cName = c['name']
                xmlChild = SubElement(xmlRoot,cName)
                self.serializeXML(c,xmlChild)
        else:
            if name == self.HISTOGRAM_ELEM:
                xmlRoot.set('n', root['count'])
                xmlRoot.tag = self.HISTOGRAM_ELEM
                for k,v in root.iteritems():
                    # guard against undefined values(these are coming from the captures
                    # in the RE_LINE_HISTO regex, and the xml module will throw exceptions on
                    # the undefined values, so we want to avoid that)
                    # and check that the key is not an internal data key
                    if v and k not in ['name','value','parent','children','colors','percentages']:
                        xmlRoot.set(k,v)

                if 'colors' in root and root['colors']:
                    strColors = ",".join(map(str,root['colors']))
                    xmlRoot.set('colors',strColors)

                if 'percentages' in root and root['percentages']:
                    strPercentages = ",".join(map(str, root['percentages']))
                    xmlRoot.set('percentages',strPercentages)

            else:
                xmlRoot.text=value

    def stripParent(self):
        """
        Returns the tree with parentless nodes.
        Note: Used for debugging purposes.
        """
        root = self.Data.copy()
        stack = []
        stack.append(root)

        while len(stack) > 0:
            x = stack.pop()
            del x['parent']
            if 'children' in x:
                stack += x['children']

        return root

    def serializeIRODS(self,root,props,parent):
        name = root['name']
        if parent:
        	name = parent+"."+name
        value = root['value']
        ret = props

        # serialize the property
        if 'children' in root and len(root['children']) > 0:
            for c in root['children']:
               	ret += self.serializeIRODS(c,props,name)
        else:
            if root['name'] == self.HISTOGRAM_ELEM:
            	# don't serialize histogram
            	pass
            else:
                ret +="%"+name+"="+json.dumps(value)
        return ret

    def toIRODS(self):
        Data = self.Data.copy()
        root = Data['children'][0]
        props = self.serializeIRODS(root,"",None)
        return props[1:] # drop the first % character

    def toJSON(self):
        """
        Returns the metadata in the JSON format
        """
        Data = self.Data.copy()
        # run transformation to compact tree
        Data = self.treeTransformCompact(Data)
        # serialize to json
        return json.dumps(Data[2], indent=2, sort_keys=True)

    def toXML(self):
        """
        Returns the metadata in the XML format
        """
        Data = self.Data.copy()
        root = Data
        # serialize the root node and return it
        tree = ElementTree(Element('Images'))
        tree.getroot().set('file',Data['children'][0]['value'])
        self.serializeXML(root, tree.getroot())
        # prettify XML and return it
        unformattedXML = tostring(tree.getroot(),encoding='utf8')
        reparsedXML = etree.fromstring(unformattedXML)
        return etree.tostring(reparsedXML, pretty_print = True)

if __name__ == '__main__':
    import argparse
    o = ImageMagickIdentifyParser()

    parser = argparse.ArgumentParser(description='ImageMagick identify -verbose parser and convertor')
    parser.add_argument("filename", help="The input file")
    parser.add_argument('--type' , '-t',default='json', help='The output type. Can be json|irods|raw|xml')
    parser.add_argument('--histo', '-H',action='store_true', help='Includes histogram section in output')
    args = parser.parse_args()


    o.optHistogram = (args.histo)
    o.parse(args.filename)

    if args.type == 'json':
        print(o.toJSON())
    elif args.type == 'irods':
        print(o.toIRODS())
    elif args.type == 'raw':
        print(o.Data)
    elif args.type == 'xml':
        print(o.toXML())
    else:
        print("Invalid type specified:" + args.type)
