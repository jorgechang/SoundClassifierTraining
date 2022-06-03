########################################################################################
###################### MAKEFILE FOR soundsource-classifierTraining MODEL #######################
########################################################################################

#####################################
## Set PROJECT_PATH AND SHELL_PROFILE
#####################################
PROJECT_PATH=${PWD}
SHELL_PROFILE=${SHELL_PROFILE_PATH}

#######
## Init
#######
ifdef SHELL_PROFILE_PATH
init: poetry paths docker
	@echo "Running shell..."
	@poetry shell
	@echo ""
else
init:
	@read -p "Your profile (.bashrc, .zshrc, .bash_profile, etc)?: " PROFILE; \
	echo "export SHELL_PROFILE_PATH='${HOME}/$$PROFILE'" >> ~/$$PROFILE; \
	echo "\033[0;33mSource your profile\033[0m";
endif

##########
## Aliases
##########
aliases:
	@echo "alias classifiertraining='make -C ${PROJECT_PATH}'" >> ${SHELL_PROFILE}

#########
## Docker
#########
docker:
	@echo "No docker services needed."

################
## Documentation
################
doc-deploy:
	@poetry run mkdocs gh-deploy --force
doc-serve:
	@poetry run mkdocs serve

######################
## Jupyter to Markdown
######################
jup2md:
ifdef jupfile
	@poetry run jupyter nbconvert ${jupfile} --to markdown
else
	@poetry run jupyter nbconvert ${PROJECT_PATH}/notebooks/examples.ipynb --to markdown
endif

#################################
## Exported environment variables
#################################
paths:
	@echo "No exported paths needed."

############################
## Poetry and ipykernel init
############################
poetry:
	@echo ""
	@echo "Installing dependencies in poetry environment..."
	@poetry install
	@echo ""
	@echo "Installing pre-commit..."
	@poetry run pre-commit install
	@echo ""
	@echo "Installing kernel..."
	@poetry run python -m ipykernel install --user --name soundsourceClassifiertraining
	@echo "\033[0;32msoundsourceClassifiertraining kernel installed.\033[0m"

#############
## Pre-commit
#############
pre-commit:
	@cd ${PROJECT_PATH}/
	@git add .
	@pre-commit run

##################
## Memory-profiler
##################
profile:
	@poetry run mprof run ${PROJECT_PATH}/profiling/classifiertraining_profiling.py > ${PROJECT_PATH}/profiling/memory_profiler.log && echo "Se creó el archivo profiling/memory_profiler.log" || echo "Error al correr memory-profiler.";
	@poetry run mprof plot -t "Recorded memory usage" -o ${PROJECT_PATH}/profiling/memory_profiler_plot.png && echo "Se creó la gráfica profiling/memory_profiler_plot.png" || echo "Error al correr memory-profiler."
	@poetry run mprof clean

############
## Streamlit
############
stream:
	@cd ${PROJECT_PATH}/app;\
		poetry run streamlit run classifiertraining_app.py

#########
## Pytest
#########
test:
	@pytest ${PROJECT_PATH}/tests/

#######
## Help
#######
help:
	@echo "#############################################################"
	@echo "##           MAKEFILE FOR soundsource-classifierTraining                 ##"
	@echo "#############################################################"
	@echo ""
	@echo "   Targets:   "
	@echo ""
	@echo "   - init: Initialize repository:"
	@echo "     - Install poetry"
	@echo "     - Install pre-commit"
	@echo "     - Install ipykernel"
	@echo "     - Check necessary paths and external dependencies"
	@echo "       Usage: % make init"
	@echo ""
	@echo "   - aliases: Create alias"
	@echo "       Usage: % make aliases"
	@echo ""
	@echo "   - doc: Deploy documentation"
	@echo "       Usage: % make doc-deploy → GitHub Page"
	@echo "       Usage: % make doc-serve  → Local"
	@echo ""
	@echo "   - jup2md: Convert Jupyter notebook to Markdown"
	@echo "       Usage: % make jup2md → Convert notebooks/examples.md"
	@echo "       Usage: % make jup2md jupfile=</path/to/jupfile>"
	@echo "                            ↳ Convert notebooks/examples.md"
	@echo ""
	@echo "   - pre-commit: Run pre-commits"
	@echo "       Usage: % make pre-commit"
	@echo ""
	@echo "   - profile: Run memory-profiler"
	@echo "       Usage: % make profile"
	@echo ""
	@echo "   - test: Run pytests"
	@echo "       Usage: % make test"
	@echo ""
	@echo "   - stream: Run streamlit app"
	@echo "       Usage: % make stream"
	@echo ""
	@echo "   - help: Display this menu"
	@echo "       Usage: % make help"
	@echo ""
	@echo "   - default: init"
	@echo ""
	@echo "   Hidden targets:"
	@echo "   "
	@echo "   - poetry"
	@echo "   "
	@echo "#############################################################"
