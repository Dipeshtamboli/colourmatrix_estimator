from Estimate_W import Wfast
import glob
import openslide
from multiprocessing import Pool
from PIL import Image
import signal


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# images = list(open("patches_path.txt"))

# images = glob.glob('/home/Drive2/cam17_patch/*/*/*/*.jpeg')
images = glob.glob('/home/Drive2/patches_256/*/*/*.jpeg')
print(len(images))
mat_w = open('patches_256.csv','w')

with Pool(initializer=initializer) as pool:
    for img_path in images:
        # img = Image.open("patches/"+img_path.strip())
        img = Image.open(img_path.strip())
        a, _ = Wfast(img,2,0.01,1,256*256,1,True,pool)
        s = f'{img_path.strip()},{a[0][0]},{a[0][1]},{a[1][0]},{a[1][1]},{a[2][0]},{a[2][1]}\n'
        mat_w.write(s)
        img.close()

# plt.subplot(321)
# sns.histplot(df[0])

# plt.subplot(322)
# sns.histplot(df[1])

# plt.subplot(323)
# sns.histplot(df[2])

# plt.subplot(324)
# sns.histplot(df[3])

# plt.subplot(325)
# sns.histplot(df[4])

# plt.subplot(326)
# sns.histplot(df[5])
