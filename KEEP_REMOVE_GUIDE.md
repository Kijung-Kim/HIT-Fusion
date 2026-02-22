# KEEP / REMOVE GUIDE

## KEEP (commit to GitHub)
- golfBall.sln
- golfBall.vcxproj
- golfBall.vcxproj.filters
- ED.cpp, EDCircles.cpp, EDColor.cpp, EDLines.cpp, EDPF.cpp, golfBall.cpp, kGolfPose.cpp, NFA.cpp, spray.cpp
- dirent.h, ED.h, EDCircles.h, EDColor.h, EDLines.h, EDPF.h, golfBall.h, kGolfPose.h, NFA.h, spray.h
- yolo.cfg
- yolo.weights (or provide download script/link)
- README.md
- .gitignore

## REMOVE from repo (generated/local artifacts)
- .vs/
- x64/
- golfBall.vcxproj.user
- UpgradeLog.htm
- bd.result*
- metrics*.csv
- generated jpg outputs (follow/impact/vote/personROI/ballYolo/back)
- local opencv dlls (optional for local testing only)
