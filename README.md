# Game-Dictionary-Word-Predictor
Word predictor based on the games dictionary that can be found [here](https://en.wikipedia.org/wiki/Glossary_of_video_game_terms)

## Run the model locally
First sync your virtual environment using the command below:
```bash
uv sync ---all-extras
```

The run the following command:
```bash
uv run src/slanggen/main.py
```

You can then open the application running the command:
```bash
uvicorn backend.app:app --reload
```

This will open up the port 8000 to open for you.

## Accessing the application
After turning on the VM UOS3-MWienhoven on SURF, you can access the VM via the following link:
```bash
http://145.38.188.207/
```

This will show the Game Word Predictor.

## Docker
The Docker image can be build via:
```bash
make build
```

The Docker image can be ran via:
```bash
make run
```

The ```make run``` command will open up the application at:
```bash
http://localhost:8000
```

The Docker image can be cleaned via:
```bash
make clean
```

## Docker Compose
To deploy the application, run the command:
```bash
make compose-up
```

This will build the image and start the service.

To stop the application, run the command:
```bash
make compose-down
```

This will shut down the running containers.

To rebuild and restart the application, run the following command:
```bash
make compose-rebuild
```

# Tutorial of Raoul: slanggen

## train the model

First, build the environment with
```bash
uv sync --all-extras
```
We use `--all-extras` because we also want to install the optional packages (`fastapi`,` beautifulsoup4`.)

and activate on UNIX systems with
```bash
source .venv/bin/activate
```

Note how I added to the pyproject.toml:
```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

This makes sure that you dont download an additional 2.5GB of GPU dependencies, if you dont need them. This can be essential in keeping the container within a manageable size.

For the Dockerfile, you can use one of my prebuilt containers:
```docker
FROM raoulgrouls/torch-python-slim:py3.12-torch2.8.0-arm64-uv0.9.8
```

This will download a small container with python 3.12, torch 2.8.0 and uv 0.9.8 installed. Check my [hub.docker](https://hub.docker.com/r/raoulgrouls/torch-python-slim/tags) for the most recent builds. If you want to build images yourself, you can use my repository [here](https://github.com/raoulg/minimal-torch-docker)

Have a look at the [slanggen.toml](slanggen.toml) file; this contains the configuration for the data and model.
Modify the url to download your own dataset; probably the scraper will work for all dictionaries on [mijnwoordenboek.nl](https://www.mijnwoordenboek.nl),  but you might need to modify the scraper if you want to use a different source. Note that you need a dataset with at least about 400 words for the model to train properly.

train the model:
```bash
python src/slanggen/main.py
```

This should fill the `assets` folder with a file `straattaal.txt` and fill the `artefacts` folder with a trained model.

# Build and publish your artefact
Design science centers around creating an **artefact**. Off course, you could argue that a jupyter notebook is an artefact too, but it has a lot of downsides in terms of reproducibility, version control and deployment. A superior way is to package your code as a wheel file that can be installed via pip or uv.

You can build the `src/slanggen` package into a wheel with `uv` very easily, you just run:
```bash
uv build --clear
```
## What is a wheel and why use it?
A Python wheel is a pre-built package format (ending in `.whl`) that contains your code and metadata in a ready-to-install format, like a zip file specifically designed for Python packages. Unlike source distributions that need to be compiled during installation, wheels are already built, making installation much faster and more reliable—especially important when you're deploying to Docker containers where you want quick, reproducible builds.

When you run `uv build --wheel`, you're creating a single portable file that bundles your entire project with all its dependencies resolved, which you can simply copy into a Docker image or upload to PyPI for others to use. This means your data science models and pipelines become distributable artifacts that anyone (including your future self) can install with a single uv (or pip) command, without worrying about compilation errors or missing dependencies.

## Publish your wheel
You could simply share your wheel file by sending it to someone, but you have been using wheels since the first moment you did `pip install package`: in that case, a wheel is downloaded from an online repository like pypi or the conda channels.

`uv build` should produce a `dist` folder, and shoud add these two files:
```bash
❯ lsd dist
.rw-r--r--@ 9.5k username  4 Dec 14:35  slanggen-0.4.tar.gz
.rw-r--r--@ 6.0k username  4 Dec 14:35  slanggen-0.4-py3-none-any.whl
```

I published slanggen at [pypi](https://pypi.org/project/slangpy/) with uv (see `uv publish --help` for more info).
You could do the same after building the wheel, making an account on [pypi.org](https://pypi.org/) and generating an API token from pypi to publish. However, it is also possible to directly install from the wheelfile; with `uv` you could do this with `uv add /path/to/slanggen-0.4-py3-none-any.whl` in a new environment. So, for example, you could:

- create a Dockerfile
- mount or copy the wheel file into the container
- inside the dockerfile, you would like to have slanggen as a package, so you can `RUN uv add /path/to/slanggen-0.4-py3-none-any.whl` pointing to the location in the container.

Or, alternatively, you could copy the wheel file to some central location where your colleagues can access it (a webserver, an S3 bucket, a file share) and install it from there with `uv add http://path/to/slanggen-0.4-py3-none-any.whl`.

# test the backend
now test the `app.py` file:

```bash
python backend/app.py
```

This will show a webpage at http://127.0.0.1:80 and you should see a blue button "generate words" and a slider for temperatur. Click the button and see if it generates some words.

# Exercise
Create one (or more) Dockerfiles inside the `straataal` directory to dockerize the backend application, such that you can run it in a docker container and deploy it on SURF.
Use your own dataset that focusses on token-level prediction (eg some dialect words, medical terms, babynames, chemical names, etc)

create Dockerfiles that:
- [ ] uses small builds (eg use my torch-python-slim images)
- [ ] installs the requirements with uv for speed
- [ ] copies all necessary backend files. Pay special attention to required paths!
- [ ] study `backend/app.py` to see what is expected
- [ ] install the slanggen from the wheel instead of copying the full src folder
- [ ] expose port 80 in the Dockerfile (this way you can access it via SURF without the need to open additional ports)

create a Makefile that:
- [ ] checks for the wheel. If the wheel doesnt exist, use `uv` to let Make automatically create it
- [ ] checks if the trained model is present. If not, late Make train the file and create the model
- [ ] builds the docker image, if the wheel and model exist.
- [ ] runs the docker on port 80
- [ ] test if you can access the application via SURF

Finally:
- [ ] implement a `docker-compose.yml` file; this way you can make sure that the service starts up automatically after you pause and resume the instance on SURF.
- [ ] publish your artefact on SURF, and hand in the URL

Optionally:
- [ ] try to improve the frontend GUI; for example, add a dropdown with starting letters, or add some nice CSS styling
- [ ] play around with your favorite dataset to generate some nice words

# Code selfstudy guide
## pyproject.toml

- check the `project.script` section. Which function is called? Check this function in the codebase.
- what should happen when you run `calculator` from the command line?
- test your answer by `cd`-ing into `3-testing`, running `uv sync` and then type `calculator` in your terminal.

## src
### __init__.py
- what do you think is the purpose of `__all__`?
- have a look at `tests/test_calculator.py`. Can you relate the imports here to the `__all__` variable?

### calculator.py and api.py
- check `calculator.py` to understand the basic calculator
- now look at `api.py` and trace back how the endpoints are using the calculator functions
- study the `/health` endpoint: what do you think is being catched here? Can you think of errors that might be missed by the healthcheck?

## tests

look at `test_calculator.py`.
    - What is being tested? And what not? First try to think about issues this approach might miss.

Now look at `test_hypothesis.py`.
- What would be the motivation for `@given`?
- Why would we need an additional check with `assume` in python? (hint: it has to do with what type the input is)
- what is being improved by the commutativity and identity tests that was not covered before?
- change the `epsilon` value in the `test_divide` function to `0` and run the tests again. What happens? Why do you think that is? Would you have found that without hypothesis?
- How would you improve the `test_add_zero` function with hypothesis?

study `test_api.py` and `test_integration.py`
- Why do you think using hypothesis in these files is NOT a good idea?


## Makefile
- trace back the dependencies of `test`; what are dependencies doing?
- `wildcard` is a Make function that will find all files matching a pattern. Can you understand what is happening here? Why would we want to have a lot of files as dependencies for the `DOCKER_ID_FILE` target?
- Do you understand the motivation for the DOCKER_ID_FILE test?

Do you understand how `@pytest.mark` interacts with `pytest -m` ?
If not, go back to the testfiles
In addition to that, check the pyproject.toml file to see how the marks are added.

# run the tests
After having studied all the components, run the tests.
First, install with uv all dependencies:
- `cd` into the 3-testing folder and run `uv sync`
- either run `make test`, or, manually build the docker image and then run the pytest commands

Study the outputs of the tests. Check the different types of tests:
1. api
1. caclulator
1. hypothesis

- What do you think about the differences that hypothesis adds?
- Why do you think hypothesis needs things like `epsilon`? If you dont understand why, set `epsilon=0` and run the tests again.

Study hypothesis statistics: why are some tests invalid? Is that an issue?

Study coverage.
- can you increase coverage by adding a new test?

# docker compose
- first do `make compose`
- then check with `docker ps` the healthcheck
- test the api manually with [http://localhost:8000/docs](http://localhost:8000/docs)

# Improve straattaal
You will improve your implementation of straattaal with tests.

## Exercise 1: Basic API Testing
Create a test file tests/test_api.py that tests the basic functionality of the FastAPI endpoints:

1. Test the /health endpoint
2. Test the /generate endpoint with different parameters:
    - Default parameters (10 words, temperature 1.0)
    - Custom number of words
    - Different temperature values
3. Test the starred words functionality:
    - Adding a word
    - Adding a duplicate word (should not duplicate)
    - Removing a word
    - Getting the starred words list

LLMs are pretty good at generating tests; just make sure you read through the tests. Go for quality over quantity. A suggestion for a prompt would be:

```
[insert backend/app.py from the straattaal repo]
I want to generate api tests for the FastAPI app above using FastAPI's TestClient.
Make sure to cover edge cases and error handling in addition to basic functionality.
Dont start testing right away; first, discuss with me what you want to test and why. Discuss your motivation, and give me the chance to learn about how you approach testing.

Only after we have discussed the approach, you can generate the tests
```

Use coverage to find untested parts of the code, and discuss these with the LLM to generate additional tests when relevant.

Learning Goals:

- Understanding API testing with FastAPI's TestClient
- Testing HTTP status codes and response content
- Testing stateful operations (starred words list)

## Exercise 2: Property-Based Testing
Create a test file tests/test_generation.py that uses Hypothesis to test the word generation functionality. Some properties to test:

- Test that generated words are always strings
- Test that the number of generated words matches the requested amount
- Test that generated words do not exceed the maximum length

Learning Goals:

- Understanding property-based testing with Hypothesis
- Testing stochastic processes
- Working with random number generators and seeds

## Exercise 3: Model and Tokenizer Testing
Create a test file tests/test_model.py that tests the model and tokenizer functionality:

Test model loading:
- Test correct loading of tokenizer
- Test correct loading of model configuration
- Test handling of missing files

Test tokenizer properties:
- The tokenizer will encode/decode. Build an additional function that uses the tokenizer to encode and decode a word, and test that the output matches the input (tokenization reversibility), i.e. word == decode(encode(word))
- if you find errors that the user could actually encounter, can you make the code more robust? E.g by adding error handling or by adding constraints to the inputs?

Learning Goals:
- Testing file operations and error handling
- Testing ML model properties
- Understanding model loading and configuration

# Exercise 4: Integration Testing with Docker
Create a test file tests/test_integration.py that tests the entire system running in Docker:

Test the complete workflow:
- Generate words
- Star some words
- Unstar words
- Verify persistence

Test error recovery:
- Invalid inputs

Learning Goals:
Understanding Docker-based testing
Error handling in a containerized environment

## Bonus Exercise: Test Coverage and Quality
- Implement pre-commit hooks for test running
- Add pytest-cov and achieve >70% test coverage

