# icecubeml

Repo containing my bachelor project (co-authored by F. Hansen).

See the pdf titled "Bachelor ... .pdf".

Main document.ipynb uses Dataloader.py and HomemadeFuncs.py to extract data from .db files (not available here) containing event data from the IceCube observatory. tcn.py is a temporal convolutional network we use for:
1. Classification of whether or not a detected muon has stopped in the detector or gone through
2. Regression of the energy of the muon
3. Regression of the zenith angle
4. Regression of the azimuth angle
