from setuptools import find_packages, setup

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> list[str]:
    '''
    This function will return the list of the requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]  # Strip any leading/trailing whitespace

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name='Online-Food-Order-Prediction',
    version='0.0.1',
    author='Tomisin',
    author_email='tomisin_adeniyi11@yahoo.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
