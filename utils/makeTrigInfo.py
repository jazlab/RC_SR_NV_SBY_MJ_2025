"""
Originally vgm_MakeTrigInfoFlex.m by Vikram Gadagkar
Ported to python by Ruidong Chen
This function creates time-aligned spike counts for each trial.
"""

import numpy as np


def _inrange(x, center, width, binsize):
    if x < center - width:
        return False
    if x > center + width + binsize:
        return False
    return True


def makeTrigInfo(
    trigger, events, offset=None, exclude=[], xl=6, binsize=0.05, s=3
):
    count = 0
    ntrials = len(trigger)
    trigEvents = [[]]
    trigStarts = [[]]
    events = np.array(events)
    edges = np.arange(-xl, xl + binsize * 2, binsize)
    histograms = []
    dist_steps_alltrials = []
    for i in np.arange(ntrials):
        tempexclude = [
            x for x in exclude if _inrange(x, trigger[i], xl, binsize)
        ]
        # if exclude is empty, then this trial is valid
        if not tempexclude:
            count = count + 1
            tempevents = events - trigger[i]
            tempevents = tempevents[tempevents <= xl + binsize]
            tempevents = tempevents[tempevents > -xl]
            if (
                offset is not None
            ):  # this is for the case when next phase bleeds into the current phase, and we need to do attrition.
                tempevents = tempevents[tempevents < offset[i]]
            trigEvents.append(tempevents)
            trigStarts.append(trigger[i])
            hist, _ = np.histogram(tempevents, edges)
            # create histogram for this one trial
            if offset is not None and offset[i] < xl:
                offset_bin = int((offset[i] + xl) / binsize)
                hist = hist.astype(float)
                hist[offset_bin:] = np.nan
            histograms.append(hist)
            # Create dist_steps_smooth
            dist_steps = []
            edges_steps = []
            for step in np.arange(-binsize / 2, binsize / 2, binsize / 10):
                dist_step, edges_step = np.histogram(tempevents, edges + step)
                if offset is not None and offset[i] + step < xl:
                    offset_bin = int((offset[i] + xl) / binsize)
                    dist_step = dist_step.astype(float)
                    dist_step[offset_bin:] = np.nan
                dist_steps.extend(dist_step)
                edges_steps.extend(edges_step[:-1])
            dist_steps = np.array(dist_steps)
            dist_steps_alltrials.append(dist_steps)

    # remove the first empty item
    trigEvents = trigEvents[1:]
    trigStarts = trigStarts[1:]
    old_way = False
    if count == 0:
        edges = []
        dist = []
        rds = []
    elif old_way:
        allevents = np.concatenate(trigEvents)
        dist, edges = np.histogram(allevents, edges)
        dist_steps = []
        edges_steps = []
        for step in np.arange(-binsize / 2, binsize / 2, binsize / 10):
            dist_step, edges_step = np.histogram(allevents, edges + step)
            dist_steps.extend(dist_step)
            edges_steps.extend(edges_step[:-1])
        dist_steps = np.array(dist_steps)
        edges_steps = np.array(edges_steps)
        # sort by edges
        idx = np.argsort(edges_steps)
        edges_steps = edges_steps[idx]
        dist_steps = dist_steps[idx]
        dist_steps = dist_steps / (count)
        dist_steps = dist_steps / binsize
        dist_steps_smooth = np.convolve(dist_steps, np.ones(s) / 3, mode="same")

        dist = dist / (count)
        dist = dist / binsize
        rds = np.convolve(dist, np.ones(s) / s, mode="same")
    else:  # new way of calculating histograms before averaging
        # Convert list of histograms to a numpy array for averaging
        histograms = np.array(histograms)
        # import matplotlib.pyplot as plt

        # plt.imshow(histograms)
        # plt.show()
        dist = np.nanmean(
            histograms, axis=0
        )  # Average across trials, ignoring nan
        rds = np.convolve(dist, np.ones(s) / s, mode="same")

        # Create dist_steps_smooth
        edges_steps = []
        for step in np.arange(-binsize / 2, binsize / 2, binsize / 10):
            _, edges_step = np.histogram(
                np.concatenate(trigEvents), edges + step
            )
            edges_steps.extend(edges_step[:-1])
        dist_steps = np.array(dist_steps_alltrials)
        edges_steps = np.array(edges_steps)
        # sort by edges
        idx = np.argsort(edges_steps)
        edges_steps = edges_steps[idx]
        dist_steps = dist_steps[:, idx]
        dist_steps = np.nanmean(dist_steps, axis=0)
        dist_steps = dist_steps / binsize
        dist_steps_smooth = np.convolve(dist_steps, np.ones(s) / 3, mode="same")
    edges = edges[:-1]

    return {
        "trigStarts": trigStarts,
        "events": trigEvents,
        "edges": edges,
        "rd": dist,
        "rds": rds,
        "edges_step": edges_steps,
        "rd_step": dist_steps,
        "rds_step": dist_steps_smooth,
    }
