class ReceiptGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        # agg_func = lambda s: [(n,w, p, t,q) for n,w, p, t,q in zip(s["receipt_number"].values.tolist(),
        #                                                        s["word"].values.tolist(),
        #                                                         s["POS"].values.tolist(),
        #                                                         s["tag"].values.tolist(),
        #                                                         s["receipt_index"].values.tolist())]
        # self.grouped = self.data.groupby("receipt_index").apply(agg_func)
        agg_func = lambda s: [(n, w, p, t, q,e,r,y,u,i,o) for n, w, p, t, q,e,r,y,u,i,o in zip(s["receipt_number"].values.tolist(),
                                                                       s["word"].values.tolist(),
                                                                       s["POS"].values.tolist(),
                                                                       s["Nalp"].values.tolist(),
                                                                       s["Nnum"].values.tolist(),
                                                                       s["Nspec"].values.tolist(),
                                                                       s["length"].values.tolist(),
                                                                       s["Ndot"].values.tolist(),
                                                                       s["Ncomma"].values.tolist(),
                                                                       s["Ncolons"].values.tolist(),
                                                                       s["tag"].values.tolist())]
        # agg_func = lambda s: [(n, w, t, q, e, r, y, u, i, o) for n, w, t, q, e, r, y, u, i, o in
        #                       zip(s["receipt_number"].values.tolist(),
        #                           s["word"].values.tolist(),
        #                           s["Nalp"].values.tolist(),
        #                           s["Nnum"].values.tolist(),
        #                           s["Nspec"].values.tolist(),
        #                           s["length"].values.tolist(),
        #                           s["Ndot"].values.tolist(),
        #                           s["Ncomma"].values.tolist(),
        #                           s["Ncolons"].values.tolist(),
        #                           s["tag"].values.tolist())]
        self.grouped = self.data.groupby("receipt_index").apply(agg_func)
        self.receipts = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["receipt_index"]
            self.n_sent += 1
            return s
        except:
            return None