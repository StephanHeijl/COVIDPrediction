import os
import camelot
import pandas

deceased_dfs = []

for pdf in sorted(os.listdir("pdfs")):
    pdf_path = os.path.join("pdfs", pdf)
    csv_fname = os.path.join("text", os.path.basename(pdf_path.rsplit(".", 1)[0]) + ".csv")
    if os.path.exists(csv_fname):
        continue
    print("Parsing %s" % pdf)
    try:
        dfs = camelot.read_pdf(
            pdf_path, flavor='stream', pages="all", quiet=True, edge_tol=500
        )
        dfs = [table.df for table in dfs]
        print("found %i tables" % len(dfs))
    except Exception as e:
        raise e
        print("found 0 tables")
        dfs = []

    for df in dfs:
        df = df.iloc[1:]
        cols = df.iloc[0]
        df.columns = cols
        df = df.iloc[1:]

        if "Datum van overlijden" in df.columns:
            print(df)
            df.set_index("Datum van overlijden", inplace=True)
            df.to_csv(csv_fname)
            deceased_dfs.append(df)
            break

