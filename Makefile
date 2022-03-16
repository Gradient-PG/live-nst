all: style type lint test

style:
	# check style
	black --check \
	--verbose \
	nst examples tests

format:
	# format all files
	black --verbose \
	nst examples tests

type:
	# check types with mypy
	mypy -p nst -p tests \
	--ignore-missing-imports \
	--show-error-context

lint:
	# lint with flake8
	flake8 nst examples tests
test:
	# run tests
	pytest tests/
