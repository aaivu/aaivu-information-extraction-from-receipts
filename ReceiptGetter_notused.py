class ReceiptGetter(object):

    def __init__(self, data):
        self.n_receipt = 1
        self.data = data
        self.empty = False

    def get_next(self):
        try:
            s = self.data[self.data["receipt_index"] == self.n_receipt]
            print((self.n_receipt))
            self.n_receipt += 1

            return s["word"].values.tolist(), s["POS"].values.tolist(), s["tag"].values.tolist()
        except:
            self.empty = True
            return None, None, None