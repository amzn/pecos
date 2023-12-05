
# Download Raw Data from NCI Paper
- NQ320K: https://drive.google.com/drive/folders/1epfUw4yQjAtqnZTQDLAUOwTJg-YMCGdD
- Trivia: https://drive.google.com/drive/folders/1SY28Idba1X8DNi4PYaDDH9CbUpdKiTXQ
```
    unzip the NQ320K folder to ./raw/NQ320K_data
    unzip the Trivia folder to ./raw/trivia_newdata
```

# Process the Raw Data to XMC Format
- NQ320K:
```
    python proc_nq320k.py
```

- Trivia:
```
    python proc_trivia.py
```

You should see the following data artifacts
```
./xmc/{nq320k|trivia}
|- X.trn.abs.txt
|- X.trn.d2q.txt
|- X.trn.doc.txt
|- X.trn.txt
|- X.tst.txt
|- Y.trn.abs.npz
|- Y.trn.d2q.npz
|- Y.trn.doc.npz
|- Y.trn.npz
|- Y.tst.npz
```
