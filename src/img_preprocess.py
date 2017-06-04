from exceptions import OSError, ValueError
from subprocess import CalledProcessError

with open(filename, "wb") as f:
    f.write(data)
    try:
        subprocess.call(['/usr/bin/mogrify', '-strip', filename])
        #Proceed to upload image, etc
    except (OSError, ValueError, CalledProcessError) as e:
        #Handle as necessary
