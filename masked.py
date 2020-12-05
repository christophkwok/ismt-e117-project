#!/usr/bin/env python3

###https://github.com/amueller/word_cloud/blob/master/examples/masked.py ###

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from PIL import Image

df = pd.read_csv("books.csv")
words = ''

for col_name, data in df.items():
    i = str(data[1])
    separate = i.split()
    for j in range(len(separate)):
        separate[j] = separate[j].lower()
    words += " ".join(separate) + " "

book_mask = np.array(Image.open("book.png"))
wc = WordCloud(background_color="black", max_words=2000, max_font_size=40, mask=book_mask, contour_width=3, contour_color='steelblue')
wc.generate(words)
wc.to_file("masked.png")

plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.figure()
plt.imshow(book_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
plt.show()
