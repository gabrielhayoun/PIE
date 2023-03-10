import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pynance",
    version="0.0.1",
    # author="I",
    # author_email="myemail@email.com",
    description="Short description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://git.git",
    # project_urls= {
    #     "Bug Tracker": "git.git/issues",
    # },
        classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    # python_requires=">=3.7",
    package_data={'package_name':  ['config/spec_train.cfg', 'config/spec_infer.cfg',
                                    'config/spec_coint.cfg', 'config/spec_crypto.cfg']}
)
