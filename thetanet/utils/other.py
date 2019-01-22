def progress_bar(progress, time):
    """ Print progress bar to console output in the format
    Progress: [#############################################     ] 90.0% in 10.22 sec

    Parameters
    ----------
    progress : float
        Value between 0 and 1.
    time : float
        Elapsed time till current progress.
    """

    print("\r   Progress: [{0:50s}] {1:.1f}% in {2:.0f} sec".format('#' * int(progress * 50),
                                                                    progress * 100, time), end='')
    if progress >= 1:
        print("\r   Progress: [{0:50s}] {1:.1f}% in {2:.2f} sec".format('#' * int(progress * 50),
                                                                        progress * 100, time))
