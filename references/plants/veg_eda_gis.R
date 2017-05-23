library(maptools)
crsveg=CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
veg = readShapePoly()
veg=st_read("/Users/jenniferwaller/galvanize/capstone/data/OSMPVegetation/OSMPVegetation.shp",options = NULL, quiet = FALSE, iGeomField = 1L,
            type = 0, promote_to_multi = TRUE,
            stringsAsFactors = default.stringsAsFactors(), int64_as_string = FALSE)
classifiers=read.csv("/Users/jenniferwaller/galvanize/capstone/data/Complete_NVCS_Summary.csv")
