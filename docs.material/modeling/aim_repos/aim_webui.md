---
title: AIM Web UI
icon: material/monitor-dashboard
---

# AIM Web UI

AIM web interface is quite intuitive and 
the official documentation already provides 
a [general purpose tutorial](https://aimstack.readthedocs.io/en/latest/ui/overview.html).

In this mini guide we limit to showcase a basic set 
of operations to navigate the ML artifacts using
some artifacts from our [IMC23](/papers/imc23) paper.

To replicate the following, make sure you [installed
the needed artifacts]().

```
aim up --repo notebooks/imc23/campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/
```

!!! info "Output"
	```
	Running Aim UI on repo `<Repo#-3653246895908991301 path=./notebooks/imc23/campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/.aim read_only=None>`
	Open http://127.0.0.1:43800
	Press Ctrl+C to exit
	```

	Run `aim up --help` for more options (e.g., specifying a different port or hostname).

When visiting the URL reported in the output 
you land on the home page of the AIM repository.

This collects a variety of aggregate metrics 
and track activity over time. 
Hence, in our scenario
the home page of the ML artifacts are mostly empty
because all campaigns were generated in a specific moment in time.

[![aim-home-page]][aim-home-page]

  [aim-home-page]: ../../figs/aim_home-page.png

The left side bar allows switch the view.
In particular, "Runs" show a tabular
view of the runs collected in the repository.

[![aim-run1]][aim-run1]

  [aim-run1]: ../../figs/aim_run1.png

From the view you can see the hash of each run
and scrolling horizontally you can glance 
over the metadata stored for each run.

[![aim-run2]][aim-run2]

  [aim-run2]: ../../figs/aim_run2.png

The search bar on the top of the page
allows to filter runs.
It accept python expression bounded
to a `run` entry point.

For instance, in the following example we filter
one specific run based on hyper parameters.

[![aim-run3]][aim-run3]

  [aim-run3]: ../../figs/aim_run3.png


!!! tip "Using the search box"
    
    The search box accept python expressions and `run.hparams` 
    is a dictionary of key-value pairs related to the different runs.

    As from the example, you can use the traditional python
    syntax of `dict[<key>] == <value>` to filter, but the search
    box supports also a dot-notated syntax `hparams.<key> == <value>`
    which has an autocomplete.

    In the example, the search is based on equality but any other
    python operation is allowed.

When clicking the hash of a run (e.g., the one we filtered)
we switch to a per-run view which
further details the collected metadata of the selected run.

[![aim-log1]][aim-log1]

  [aim-log1]: ../../figs/aim_log1.png

For instance, when scrolling at
the bottom of the per-run page
we can see that AIM details

* The specific git commit used when executing the run.

* The specific python packages and related versions
available in the environment when executing the run.

Both are automatically tracked by AIM with
no extra code required (beside activating the 
their collection when creating the run).

[![aim-log2]][aim-log2]

  [aim-log2]: ../../figs/aim_log2.png

The per-run view offers a variety of information
organized in multiple tabs.

For instance, the tab "Logs"
details the console output.

[![aim-log3]][aim-log3]

  [aim-log3]: ../../figs/aim_log3.png

