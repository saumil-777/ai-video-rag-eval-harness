import os
import sys
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


def install_ragas_bridge():
    """Installs the langchain_community.chat_models.vertexai compatibility bridge

    into site-packages so standalone `import ragas` works out-of-the-box
    in any fresh environment after running `pip install -r Requirements.txt`.
    """
    try:
        import site

        site_packages_dirs = site.getsitepackages()
        if hasattr(site, "getusersitepackages"):
            site_packages_dirs.append(site.getusersitepackages())

        for sp in site_packages_dirs:
            target_dir = os.path.join(sp, "langchain_community", "chat_models")
            if os.path.exists(target_dir):
                vertexai_file = os.path.join(target_dir, "vertexai.py")
                with open(vertexai_file, "w", encoding="utf-8") as f:
                    f.write(
                        "# Compatibility bridge for Ragas 0.4.x / LangChain 0.4.x VertexAI import boundary\n"
                        "try:\n"
                        "    from langchain_google_vertexai import ChatVertexAI\n"
                        "except Exception:\n"
                        "    from langchain_core.language_models.chat_models import BaseChatModel as ChatVertexAI\n\n"
                        '__all__ = ["ChatVertexAI"]\n'
                    )
                print(f"Installed Ragas compatibility bridge at {vertexai_file}")
    except Exception as e:
        print(f"Warning: Could not install Ragas bridge automatically: {e}")


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        install_ragas_bridge()


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        install_ragas_bridge()


# Run bridge install during direct setup.py execution
install_ragas_bridge()

setup(
    name="ai-video-assistant",
    version="1.0.0",
    description="AI Video Assistant RAG Application",
    py_modules=[],
    cmdclass={
        "develop": CustomDevelopCommand,
        "install": CustomInstallCommand,
    },
)
