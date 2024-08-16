from sklearn.datasets import fetch_20newsgroups
train_dataset = fetch_20newsgroups(subset='train', categories = categories, remove=('headers', 'footers', 'quotes'), random_state=0)
