import os
import matplotlib.pyplot as plt
from PIL import Image

"""
location = "tajmahal"  
BASE_DIR = f"models/TrainedModels/{location}/scale_factor=0.750000,alpha=10"
scales = [1, 3, 5, 7]

fig, axes = plt.subplots(
    nrows=2,
    ncols=len(scales),
    figsize=(4 * len(scales), 6)
)

for col, scale in enumerate(scales):
    scale_dir = os.path.join(BASE_DIR, str(scale))

    real_img = Image.open(os.path.join(scale_dir, "real_scale.png"))
    fake_img = Image.open(os.path.join(scale_dir, "fake_sample.png"))

    axes[0, col].imshow(real_img)
    axes[0, col].set_title(f"Scale {scale}", fontsize=14)
    axes[0, col].axis("off")

    axes[1, col].imshow(fake_img)
    axes[1, col].axis("off")


fig.text(0.08, 0.65, "Real", fontsize=16, va="center", ha="left")
fig.text(0.08, 0.25, "Fake", fontsize=16, va="center", ha="left")


output_name = f"summary_scales_real_vs_fake_circular_padding{location}.png"
plt.savefig(output_name, dpi=300, bbox_inches="tight")
plt.close()
"""




def plot_real_vs_two_models(dataset="etretat", models=["Frechet", "NNPL"], scales=[1,3,5,7]):
    
    #Plot real images and fake_samples from two models across scales.
    
    n_rows = 3  # Real, model1, model2
    n_cols = len(scales)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4*n_cols, 3*n_rows))

    for c_idx, scale in enumerate(scales):
        real_dir = f"models/TrainedModels_{models[0]}/{dataset}/scale_factor=0.750000,alpha=10/{scale}"
        real_img = Image.open(os.path.join(real_dir, "real_scale.png"))
        axes[0, c_idx].imshow(real_img)
        axes[0, c_idx].set_title(f"Scale {scale}", fontsize=12)
        axes[0, c_idx].axis("off")

        for r_idx, model in enumerate(models, start=1):
            model_dir = f"models/TrainedModels_{model}/{dataset}/scale_factor=0.750000,alpha=10/{scale}"
            fake_img = Image.open(os.path.join(model_dir, "fake_sample.png"))
            axes[r_idx, c_idx].imshow(fake_img)
            axes[r_idx, c_idx].axis("off")

    fig.text(0.07, 0.75, "Real", fontsize=14, va="center", ha="left")
    fig.text(0.07, 0.45, models[0], fontsize=14, va="center", ha="left")
    fig.text(0.07, 0.15, models[1], fontsize=14, va="center", ha="left")

    output_name = f"summary_scales_real_vs_models_{dataset}.png"
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close()


plot_real_vs_two_models(dataset="tajmahal", models=["Frechet", "NNPL"], scales=[1,3,5,7])



