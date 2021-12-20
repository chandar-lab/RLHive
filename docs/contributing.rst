Contributing
============

Core RLHive
-----------
Did you spot a bug, or is there something that you think should be added to the main 
RLHive package? Check our `Issues <https://github.com/chandar-lab/RLHive/issues>`_ on
Github or our :ref:`roadmap <roadmap>` to see if it's already on our radar. If not, 
you can either open an issue to let us know, or you can fork our repo 
and create a pull request with your own feature/bug fix. 

Creating Issues
---------------
We'd love to hear from you on how we can improve RLHive! When creating issues, please
follow these guidelines:

* Tag your issue with a bug, feature request, or question to help us effectively 
  sort through the issues.
* Include the version of RLHive you are running (run 
  ``pip list | grep rlhive``)


Creating PRs
------------
When contributing to RLHive, please follow these guidelines:

* Create a separate PR for each feature you are adding. 
* When you are done writing the feature, create a pull request to the dev branch of the
  `main repo <https://github.com/chandar-lab/RLHive>`_.
* Each pull request must pass all unit tests and linting checks before being merged. 
  Please run these checks locally before creating PRs/pushing changes to the PR to
  minimize unnecessary runs on our CI system. You can run the following commands from
  the root directory of the project:

  * Unit Tests: ``python -m pytest tests/`` 
  * Linter: ``black .`` 
  
* Information (such as installation instructions and editor integrations) for the 
  `black <https://black.readthedocs.io/>`_ formatter 
  is available here. 
* Make sure your code is documented using Google style docstrings. For examples, see
  the rest of our repository.

Contrib RLHive
--------------
We want to encourage users to contribute their own custom components to RLHive. This
could be things like agents, environments, runners, replays, optimizers, and anything
else. This will allow everyone to easily use and build on your work. 

To do this, we have a ``contrib`` directory that will be part of the package. After
adding new components and any relevant citation information to this folder, new
versions of RLHive will be updated with these components, allowing your work
to get a potentially larger audience. Note, we will be actively maintaining only
the code in the main package, not the ``contrib`` package. We can only commit to giving
minimal feedback during the review stage. If a contribution becomes widely adopted
by the community, we may move it to the main repository to actively maintain.

When submitting the PR for your contributions, you must provide some results with your
new components **that were generated with the RLHive package**, to provide us evidence
of the correctness of your implementation.
