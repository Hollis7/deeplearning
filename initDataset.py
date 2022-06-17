def initfile(oldfile, newfile):
    with open(oldfile, "r") as f:
        rn = f.readlines()
        j=1
        for i in rn:
            with open(newfile, "a") as nf:
                nf.write(str(j))
                if (j <= 50):
                    nf.write(",0")
                elif 100 >= j > 50:
                    nf.write(",1")
                elif j > 100:
                    nf.write(",2")
                nf.write("," + i)
            j=j+1


if __name__ == "__main__":
    initfile(r'data/iris.data', r'data/newiris.data')
    print("success")
