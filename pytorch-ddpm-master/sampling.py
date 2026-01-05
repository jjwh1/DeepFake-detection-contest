import os
import argparse
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from model import UNet
from tqdm import tqdm
from tqdm import trange



# ---------------------------------------------------------
# Inverse Fusion Sampler (Algorithm 1)
# ---------------------------------------------------------
@torch.no_grad()
def inverse_fusion_sampling(
    model,
    x_m,
    mask,
    *,
    T,
    n_inv,
    beta_1,
    beta_T,
    device
):
    B, C, H, W = x_m.shape

    betas = torch.linspace(beta_1, beta_T, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    def extract(v, t, shape):
        return v[t].view(-1, 1, 1, 1).expand(shape)

    # x_T ~ N(0, I)
    x_t = torch.randn((B, C, H, W), device=device)

    for t in trange(T-1, -1, -1, desc="Diffusion steps", leave=False):

        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        eps_theta = model(x_t, t_tensor)

        a_bar_t = extract(alpha_bar, t_tensor, x_t.shape)
        sqrt_a_bar = torch.sqrt(a_bar_t)
        sqrt_one_minus = torch.sqrt(1.0 - a_bar_t)

        # x̄₀
        x0_bar = (x_t - sqrt_one_minus * eps_theta) / sqrt_a_bar

        # mask fusion
        x0_tilde = mask * x0_bar + x_m

        # inverse fusion
        if t > 0:
            for _ in range(n_inv):
                eps = torch.randn_like(x_t)
                x_t = sqrt_a_bar * x0_tilde + sqrt_one_minus * eps
        else:
            x_t = x0_tilde

        # posterior sampling
        if t > 0:
            a_t = alphas[t]
            a_bar_prev = alpha_bar[t - 1]

            coef1 = torch.sqrt(a_bar_prev) * (1 - a_t) / (1 - a_bar_t)
            coef2 = torch.sqrt(a_t) * (1 - a_bar_prev) / (1 - a_bar_t)

            mu = coef1 * x0_tilde + coef2 * x_t
            sigma = torch.sqrt(betas[t])
            z = torch.randn_like(x_t)

            x_t = mu + sigma * z
        else:
            x_t = x0_tilde

    return x_t


# ---------------------------------------------------------
# Image loading
# ---------------------------------------------------------
def load_image(path, img_size, grayscale=False):
    tfs = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
    if not grayscale:
        tfs.append(transforms.Normalize([0.5]*3, [0.5]*3))
    tf = transforms.Compose(tfs)

    img = Image.open(path)
    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    img = tf(img)
    return img


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # ---------------------------
    # model
    # ---------------------------
    model = UNet(
        T=args.T_model,
        ch=args.ch,
        ch_mult=args.ch_mult,
        attn=args.attn,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["ema_model"])
    model.eval()

    # ---------------------------
    # data
    # ---------------------------
    input_files = []

    for d in args.input_dirs:
        if not os.path.isdir(d):
            print(f"[!] Skip invalid dir: {d}")
            continue

        files = sorted(os.listdir(d))
        files = [os.path.join(d, f) for f in files]
        input_files.extend(files)

    print(f"[INFO] Total inference images: {len(input_files)}")

    pbar = tqdm(input_files, desc="Processing images", ncols=100)

    for img_path in pbar:
        name = os.path.basename(img_path)

        x_0 = load_image(
            img_path,
            args.img_size
        )

        mask_path = os.path.join(args.mask_dir, name)
        if not os.path.exists(mask_path):
            pbar.set_postfix_str(f"mask missing: {name}")
            continue

        m = load_image(
            mask_path,
            args.img_size,
            grayscale=True
        )

        x_0 = x_0.unsqueeze(0).to(device)
        m = (m > 0.5).float().unsqueeze(0).to(device)
        m = m.expand_as(x_0)

        x_m = x_0 * (1 - m)

        x0 = inverse_fusion_sampling(
            model,
            x_m,
            m,
            T=args.T_sample,
            n_inv=args.n_inv,
            beta_1=args.beta_1,
            beta_T=args.beta_T,
            device=device
        )

        x0 = (x0 + 1) / 2
        save_image(x0, os.path.join(args.save_dir, name))

        # ✔ 한 장 끝날 때만 상태 갱신
        pbar.set_postfix_str(f"saved={name}")


        
    
    
    
    


# ---------------------------------------------------------
# Argparser
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inverse Fusion Diffusion Inference")

    parser.add_argument("--ckpt", type=str, default='/content/drive/MyDrive/IITD/I3FDM_Distance_db1/ckpt_last.pt')
    # parser.add_argument("--input_dir", type=str, default=r"C:\Users\8138\Desktop\DB\CASIAv4\CASIA_Iris_Lamp\reflection_random(50to1.7)_db1_224")
    # parser.add_argument("--input_dir", type=str, default=r"C:\Users\8138\Desktop\DB\UPOL\reflection_random(50to1.7)_db1_224")
    parser.add_argument(
    "--input_dirs",
    nargs="+",
    type=str,
    default=[
        '/content/dataset/reflection_random(50to1.7)_db1_224_trainset',
        '/content/dataset/reflection_random(50to1.7)_db1_224_validset',
    ],
    help="Input directories for inference (train + valid)"
    )

    parser.add_argument("--mask_dir", type=str,default='/content/dataset/CASIA_Distance/algorithm/450to50000_174x174padding_if_gac1_4000_algorithm/db1_test_layer12_0.3_only_mask_h2.8_w3')
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/IITD/I3FDM_Distance_db1/test_db1_largemask_T_sample_1000_betaT_0.01")

    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--T_model", type=int, default=1000)  # 고정
    # parser.add_argument("--T_sample", type=int, default=50)
    parser.add_argument("--T_sample", type=int, default=1000)
    parser.add_argument("--n_inv", type=int, default=10)

    parser.add_argument("--beta_1", type=float, default=1e-4)
    # parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--beta_T", type=float, default=0.01)

    # UNet params (train과 동일)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--ch_mult", nargs="+", type=int, default=[1, 2, 2, 2])
    parser.add_argument("--attn", nargs="+", type=int, default=[1])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()
    main(args)
