import numpy as np
import matplotlib.pyplot as plt


def pad_image(image, value):
    return np.pad(image, ((1, 1), (1, 1), (0, 0)), constant_values=value)


def compute_image_contributions(image, inf_value=256, dbg=False):
    # pad image
    image = pad_image(image, inf_value)

    # compute contributions of all cells,
    # starting from bottom left
    # uses lowert star filtration

    contributions = dict()

    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            # lets track all the contributions
            # from cell i,j

            # itself, 2d cell
            f = tuple(image[i, j])
            contributions[f] = contributions.get(f, 0) + 1

            # 0d cell SW
            f = tuple(
                np.fmin(
                    image[i, j],
                    np.fmin(
                        image[i - 1, j - 1], np.fmin(image[i - 1, j], image[i, j - 1])
                    ),
                )
            )
            contributions[f] = contributions.get(f, 0) + 1

            # 1d cell W
            f = tuple(np.fmin(image[i, j], image[i, j - 1]))
            contributions[f] = contributions.get(f, 0) - 1

            # 1d cell S
            f = tuple(np.fmin(image[i, j], image[i - 1, j]))
            contributions[f] = contributions.get(f, 0) - 1

            if dbg:
                print(i, j, contributions)

    # remove contributions at infinity
    infinity = tuple(np.array([inf_value for i in range(image.shape[-1])]))
    del contributions[infinity]

    # remove the contributions that are 0
    to_del = []
    for key in contributions:
        if contributions[key] == 0:
            to_del.append(key)
    for key in to_del:
        del contributions[key]

    return sorted(list(contributions.items()), key=lambda x: x[0])


def EC_at_bifiltration(contributions, f1, f2):
    return sum([c[1] for c in contributions if (c[0][0] <= f1) and (c[0][1] <= f2)])


def plot_2d_ECP(contributions, limits, this_ax=None, colorbar=False, **kwargs):

    f1min, f1max, f2min, f2max = limits

    if this_ax == None:
        this_ax = plt.gca()

    f1_list = [f1min] + sorted(set([c[0][0] for c in contributions])) + [f1max]
    f2_list = [f2min] + sorted(set([c[0][1] for c in contributions])) + [f2max]

    Z = np.zeros((len(f2_list) - 1, len(f1_list) - 1))

    for i, f1 in enumerate(f1_list[:-1]):
        for j, f2 in enumerate(f2_list[:-1]):
            Z[j, i] = EC_at_bifiltration(contributions, f1, f2)

    # Plotting
    im = this_ax.pcolormesh(f1_list, f2_list, Z, **kwargs)

    if colorbar:
        plt.colorbar(im, ax=this_ax)

    return this_ax


def prune_contributions(contributions):

    total_ECP = dict()

    for a in contributions:
        total_ECP[a[0]] = total_ECP.get(a[0], 0) + a[1]

    # remove the contributions that are 0
    to_del = []
    for key in total_ECP:
        if total_ECP[key] == 0:
            to_del.append(key)
    for key in to_del:
        del total_ECP[key]

    return sorted(list(total_ECP.items()), key=lambda x: x[0])


def difference_ECP(ecp_1, ecp_2, return_contributions=False):
    fmin = 0
    fmax = 257

    contributions = [((fmin, fmin), 0), ((fmax, fmax), 0)]

    contributions += ecp_1
    contributions += [(c[0], -1 * c[1]) for c in ecp_2]

    contributions = (
        [((fmin, fmin), 0)] + prune_contributions(contributions) + [((fmax, fmax), 0)]
    )

    #     print(contributions)

    R_list = sorted(set([c[0][0] for c in contributions]))
    G_list = sorted(set([c[0][1] for c in contributions]))

    difference = 0

    for i, r in enumerate(R_list[:-1]):
        delta_r = R_list[i + 1] - R_list[i]
        for j, g in enumerate(G_list[:-1]):
            delta_g = G_list[j + 1] - G_list[j]

            difference += abs(EC_at_value(contributions, r, g) * delta_r * delta_g)

    if return_contributions:
        return difference, contributions
    else:
        return difference
