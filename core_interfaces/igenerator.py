"""
interface for GAN generators
"""

class IGenerator:
    """
    Interface for GAN generators.
    """

    def generate(self, labels: list[int]) -> list:
        """
        Generate a specified number of samples.

        :param num_samples: The number of samples to generate.
        :return: A list of generated samples.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @property
    def get_name(self) -> str:
        """
        Get the name of the generator.

        :return: The name of the generator.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")