# Arquivo de teste para verificar Ruff


def test_function(x, y):
    """Função de teste mal formatada."""
    result = x + y
    if result > 10:
        print("Resultado maior que 10")
    return result


class TestClass:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name
