all: style type lint test

style:
	# check style
	black --check \
	--verbose \
	nst

format:
	# format all files
	black --verbose \
	nst

type:
	# check types with mypy
	mypy -p nst \
	--ignore-missing-imports \
	--show-error-context

lint:
	# lint with flake8
	flake8 nst
test:
	# run tests
	pytest tests/
