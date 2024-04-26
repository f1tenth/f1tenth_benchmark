def get_git_info() -> dict[str, str]:
    """
    Extract repository information from the git repository.
    """
    import git

    repo = git.Repo(search_parent_directories=True)
    reponame = repo.remotes.origin.url.split("/")[-1].split(".")[0]
    sha = repo.head.object.hexsha
    branch = repo.active_branch.name
    return {"repo": reponame, "commit": sha, "branch": branch}


def scatter_point_on_track(track, poses):
    import matplotlib.pyplot as plt

    map = track.occupancy_map
    raceline = track.raceline

    xs, ys = raceline.xs, raceline.ys
    origin = track.spec.origin
    resolution = track.spec.resolution

    pxs = (xs - origin[0]) / resolution
    pys = (ys - origin[1]) / resolution

    plt.imshow(map, cmap="gray")
    plt.plot(pxs, pys, "r")

    pxs = (poses[:, 0] - origin[0]) / resolution
    pys = (poses[:, 1] - origin[1]) / resolution
    plt.plot(pxs, pys, "go")
