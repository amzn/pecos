# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment


## Contributing via Pull Requests
Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the `mainline` branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted. If there is an existing issue your are interested in working on, you should comment on it so others don't start working on the same issue.

To send us a pull request, please:

1. Create a personal fork of the project on Github webpage. Clone the fork on your local machine.
2. Add the original repository as a remote called `upstream`. Your remote repo on Github is called `origin`.
    ```
    git remote add upstream https://github.com/amzn/pecos.git
    git remote -v # Display all remotes to double-check
    ```
3. If you created your fork a while ago, be sure to pull upstream changes into your local repository.
    ```
    git checkout mainline
    git pull -r upstream mainline
    ```
4. Create a new branch from `mainline` to work on, or rebase your working branch on newest `mainline`.
    ```
    git checkout -b <BRANCH NAME>
    ```
5. Implement your code on the new branch:
    * Follow the code style of the project.
    * Write or adapt tests as needed.
    * Add or change the documentation as needed.
6. **Ensure local style/type checks and tests pass.** First ensure you install the following for style-checking and unit-testing
    ```
	python3 -m pip install flake8 black mypy
    ```
	Then you can use the `Makefile` commands to check:
    ```
    make clean
    make format --keep-going
    make test
    ```
    Pass `VFLAG=-v` environment variable can trigger verbose mode for further debugging:
    ```
    make clean VFLAG=-v
    make format VFLAG=-v --keep-going
    make test VFLAG=-v
    ```
    Remember using `make clean` to clean up local build after code change and before unit test.
7. Commit using clear messages. **Squash your commits into a single commit.**
8. Push your branch to **your fork** `origin` on Github.
    ```
    git push --set-upstream origin <BRANCH NAME>
    ```
9. On GitHub webpage's pull request panel, open a pull request into **original repository** `upstream` from **your fork**. *(NOTE: this is the default option if one does not make changes when creating PR)*
    * Carefully fill out PR template
    * Click on "Draft PR" on drop-down menu to double-check by oneself
    * When ready, click on “Ready for review”
10. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.
12. If you need to make changes to the pull request, simply modify on your branch by **amend** the commit, and then *force-push* to your `origin` branch. The pull request will automatically update its timeline:
    ```
    # Do modification
    git commit -a --amend
    git push -f
    ```
11. Once the pull request is approved, do the following:

    a. Check if your branch need updates from newest `mainline`. If so, click on the button and GitHub will automatically update for you.

    b. Click on the draw-down menu, click on **squash-and-merge** button to merge the pull request.

12. Pull the changes from upstream to your local repo and delete your extra branch(es).

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).


## Finding contributions to work on
Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'Call for Contribution' or 'good first issue' issues is a great place to start.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.
