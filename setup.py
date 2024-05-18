from setuptools import setup, find_packages

from project_metadata import NAME, VERSION, AUTHOR, DESCRIPTION, LONG_DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email="hello@phaseai.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "phasellm>=0.0.22",
        "Django>=5.0.0",
        "python-dotenv>=1.0.0",
        "dateparser>=1.2.0",
        "pytest-playwright",
        "beautifulsoup4",
        "chromadb",
        "feedparser",
        "pypdf",
        "faiss-cpu",
        "microsoft-bing-newssearch",
        "scrapingbee",
        "tiktoken",
    ],
    extras_require={
        "docs": [
            "furo",
            "sphinx>=7.1.2",
            "myst_parser>=2.0.0",
            "sphinx-autoapi>=2.1.1",
            "sphinx-autobuild>=2021.3.14",
        ]
    },
    python_requires=">=3.10.0",
    keywords="llm, nlp, ai, social, politics, economics",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
