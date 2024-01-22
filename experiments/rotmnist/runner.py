import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from jax import random
from smallnet import smallnet
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from augerino import datasets, losses, models
from augerino.configs.pgm_mnist import get_config
from augerino.inv.input import get_data


def main(args):
    with wandb.init(
        mode="online",
        project="learning-invariances",
        entity="shreyaspadhy",
        name="",
    ) as run:
        net = smallnet(in_channels=1, num_targets=10)
        augerino = models.UniformAug()
        model = models.AugAveragedModel(net, augerino, ncopies=args.ncopies)
        init_model = models.AugAveragedModel(net, augerino, ncopies=args.ncopies)

        start_widths = torch.ones(6) * -5.0
        start_widths[2] = 1.0
        model.aug.set_width(start_widths)

        softplus = torch.nn.Softplus()

        rng = random.PRNGKey(0)
        (
            data_rng,
            proto_init_rng,
            gen_init_rng,
            proto_state_rng,
            gen_state_rng,
        ) = random.split(rng, 5)

        config = get_config(f"{args.angle}")

        config.batch_size = args.batch_size
        config.num_epochs = args.epochs
        train_ds, val_ds, _ = get_data(config, data_rng)

        # dataset = datasets.RotMNIST("~/datasets/", train=True)
        # trainloader = DataLoader(dataset, batch_size=args.batch_size)

        optimizer = torch.optim.Adam(
            [
                {
                    "name": "model",
                    "params": model.model.parameters(),
                    "weight_decay": args.wd,
                },
                {"name": "aug", "params": model.aug.parameters(), "weight_decay": 0.0},
            ],
            lr=args.lr,
        )
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            model = model.cuda()
            print("Using Cuda")

        ## save init model ##
        os.makedirs(args.dir, exist_ok=True)
        init_fname = "/model" + str(args.aug_reg) + "_" + str(args.angle) + "_init.pt"
        torch.save(model.state_dict(), args.dir + init_fname)

        criterion = losses.safe_unif_aug_loss
        logger = []
        for epoch in tqdm(range(1)):  # loop over the dataset multiple times
            epoch_loss = 0
            batches = 0
            print("num steps : ", len(train_ds))
            for i, data in tqdm(enumerate(iter(train_ds), 0)):
                # for i, data in tqdm(enumerate(trainloader, 0)):
                # get the inputs; data is a list of [inputs, labels]

                # inputs, labels = data
                inputs, labels = data["image"].numpy(), data["label"].numpy()

                inputs = np.transpose(np.squeeze(inputs, axis=0), (0, 3, 1, 2))
                labels = np.squeeze(labels, axis=0)

                inputs = torch.tensor(inputs).float()
                labels = torch.tensor(labels).long()

                if use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                # print(inputs.shape)
                outputs = model(inputs)
                loss = criterion(outputs, labels, model, reg=args.aug_reg)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
                batches += 1
                # print(epoch, loss.item(), softplus(model.aug.width).detach().data)
                log = model.aug.width.tolist()
                log += model.aug.width.grad.data.tolist()
                log += [loss.item()]
                logger.append(log)

                run.log(
                    {
                        "loss": loss.item(),
                        "aug_width": wandb.Histogram(
                            softplus(model.aug.width).detach().data.tolist()
                        ),
                        "grad_aug_width": wandb.Histogram(
                            softplus(model.aug.width.grad.data).detach().data.tolist(),
                        ),
                    }
                )
            run.log({"epoch_loss": epoch_loss / batches, "epoch": epoch})

        fname = "/model" + str(args.aug_reg) + "_" + str(args.angle) + ".pt"
        torch.save(model.state_dict(), args.dir + fname)
        df = pd.DataFrame(logger)
        df.to_pickle(
            args.dir + "/auglog_" + str(args.aug_reg) + "_" + str(args.angle) + ".pkl"
        )

        ######## Do Eval ########
        dataset = datasets.RotMNIST("~/datasets/", train=False)
        testloader = DataLoader(dataset, batch_size=128)
        testimg, testlab = next(iter(testloader))

        ind = 64

        n_ang = 50
        angles = torch.linspace(-np.pi, np.pi, n_ang)

        four = testimg[ind, ::].unsqueeze(0)
        batch_four = torch.cat(n_ang * [four])

        with torch.no_grad():
            # Build affine matrices for random translation of each image
            affineMatrices = torch.zeros(n_ang, 2, 3)
            affineMatrices[:, 0, 0] = angles.cos()
            affineMatrices[:, 1, 1] = angles.cos()
            affineMatrices[:, 0, 1] = angles.sin()
            affineMatrices[:, 1, 0] = -angles.sin()

            flowgrid = F.affine_grid(affineMatrices, size=batch_four.size())

            rot_four = F.grid_sample(batch_four, flowgrid)

        model.eval()
        sftmx = torch.nn.Softmax(-1)

        with torch.no_grad():
            # low_preds = low_model(rot_four)

            # normalise rot_four to [-1, 1]
            rot_four_new = (rot_four - rot_four.min()) / (
                rot_four.max() - rot_four.min()
            )
            rot_four_new = 2 * rot_four_new - 1
            mid_preds = model(rot_four_new.cuda())
            mid_probs = sftmx(mid_preds.cpu())

        tick_pts = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
        tick_labs = [r"-$\pi$", r"-$\pi$/2", "0", r"$\pi$/2", r"$\pi$"]

        softplus = torch.nn.Softplus()
        test_pts = torch.linspace(-2, 2, 10)

        def get_density(test_pts, width):
            dist = torch.distributions.Uniform(-width / 2.0, width / 2.0)

            dens = []
            for test_pt in test_pts:
                if test_pt < -width / 2.0 or test_pt > width / 2.0:
                    dens.append(torch.tensor(0.0))
                else:
                    dens.append(dist.log_prob(test_pt).exp().detach())

            dens = torch.stack(dens)
            return dens

        init_test_pts = torch.linspace(
            -softplus(torch.tensor(-1.0)) / 2, softplus(torch.tensor(-1.0)) / 2, 10
        )

        width = min(softplus(model.aug.width[2].cpu()), 2 * np.pi)
        print(f"width: {width}, {softplus(model.aug.width[2].cpu())}")

        run.log({"width": softplus(model.aug.width[2].cpu())})

        init_dens = get_density(init_test_pts, softplus(torch.tensor(-1.0)))
        mid_dens = get_density(test_pts, width)

        def plot_img(fig, img, i, j, label):
            ax = plt.Subplot(fig, inner[i, j])
            ax.imshow(img, cmap="Greys", interpolation="nearest")
            ax.set_title(label, fontsize=ax_fs + 4, y=-0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

        import matplotlib.gridspec as gridspec

        sns.set_style("white")
        fig = plt.figure(figsize=(20, 4), dpi=200)
        outer = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.2)
        alpha = 0.5
        ax_fs = 18
        lwd = 4.0
        ## plot mario and iggy

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=outer[0], wspace=0.1, hspace=0.25
        )
        ind = 42
        plot_img(fig, testimg[ind, 0, ::].cpu().detach(), 0, 0, "")
        plot_img(fig, testimg[ind + 1, 0, ::].cpu().detach(), 0, 1, "")
        plot_img(fig, testimg[ind + 2, 0, ::].cpu().detach(), 0, 2, "")
        plot_img(fig, testimg[ind + 3, 0, ::].cpu().detach(), 1, 0, "")
        plot_img(fig, testimg[ind + 4, 0, ::].cpu().detach(), 1, 1, "Data Samples")
        plot_img(fig, testimg[ind + 5, 0, ::].cpu().detach(), 1, 2, "")
        ax = plt.Subplot(fig, inner[1, 1])

        ## Plot Learned Distribution
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 1, subplot_spec=outer[1], wspace=0.1, hspace=0.1
        )
        ax = plt.Subplot(fig, inner[0])
        # ax.plot(test_pts, low_dens.detach().cpu(), linewidth=lwd, label="No Reg.",
        #         alpha=alpha, linestyle="-")
        ax.plot(
            test_pts,
            mid_dens.detach().cpu(),
            linewidth=lwd,
            label="Std. Reg.",
            alpha=alpha,
            linestyle="--",
        )
        ax.plot(
            init_test_pts,
            init_dens.detach(),
            linewidth=lwd,
            label="Initial",
            color="gray",
            alpha=0.7,
        )
        ax.legend(
            loc="lower left", fontsize=ax_fs - 1, ncol=2, bbox_to_anchor=(0.02, -0.5)
        )
        ax.set_xlabel("Rotation", fontsize=ax_fs)
        ax.set_ylabel("Probability", fontsize=ax_fs)
        tick_pts = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
        tick_labs = [r"-$\pi$/2", r"-$\pi$/4", "0", r"$\pi$/4", r"$\pi$/2"]
        ax.set_xticks(tick_pts)
        ax.set_xticklabels(tick_labs)
        ax.set_xlim(-np.pi / 2, np.pi / 2)
        ax.tick_params("both", labelsize=ax_fs - 2)
        sns.despine(ax=ax)
        fig.add_subplot(ax)

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 1, subplot_spec=outer[2], wspace=0.1, hspace=0.1
        )
        ax = plt.Subplot(fig, inner[0])
        alpha = 0.7
        num = 4
        lwd = 4
        # ax.plot(angles, low_probs[:, num].detach().cpu(), linewidth=lwd, label="Low Reg.",
        #         alpha=alpha, linestyle="-")
        ax.plot(
            angles,
            mid_probs[:, num].detach().cpu(),
            linewidth=lwd,
            label="Augerino",
            alpha=alpha,
            linestyle="-",
        )
        # ax.plot(angles, e2_probs[:, num].detach().cpu(), linewidth=3, label="E2",
        #         color='gray', alpha=0.6)
        ax.set_ylim(0.0, 1.001)
        ax.set_xlabel("Rotation of Input", fontsize=ax_fs)
        ax.set_ylabel("Predicted Probability", fontsize=ax_fs)
        tick_pts = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
        tick_labs = [r"-$\pi$", r"-$\pi$/2", "0", r"$\pi$/2", r"$\pi$"]
        ax.tick_params("both", labelsize=ax_fs - 2)
        ax.set_xticks(tick_pts)
        ax.set_xticklabels(tick_labs)

        # ax.set_yticks([0.98, 0.99, 1.])
        ax.set_xlim(-np.pi, np.pi)
        ax.legend(
            loc="lower left", fontsize=ax_fs - 1, ncol=2, bbox_to_anchor=(0.02, -0.54)
        )
        sns.despine(ax=ax)
        fig.add_subplot(ax)
        plt.savefig("./rotmnist_full.pdf", bbox_inches="tight")
        run.log({"rotmnist_full": wandb.Image(plt)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="olivetti augerino")

    parser.add_argument(
        "--dir",
        type=str,
        default="./saved-outputs",
        help="training directory (default: None)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        metavar="N",
        help="input batch size (default: 50)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--aug_reg",
        type=float,
        default=0.1,
        help="augmentation regularization weight",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4,
        metavar="weight_decay",
        help="weight decay",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=75,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )

    parser.add_argument(
        "--ncopies",
        type=int,
        default=4,
        metavar="N",
        help="number of augmentations in network (defualt: 4)",
    )

    parser.add_argument(
        "--angle",
        type=int,
        default=30,
        metavar="N",
        help="max angle of rotation in training set (defualt: 30)",
    )
    args = parser.parse_args()

    main(args)
