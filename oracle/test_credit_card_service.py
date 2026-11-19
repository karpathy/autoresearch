import requests
import logging

# Constantes de URL
_ASSOCIATE_URL = "http://localhost:{port}/associate_card"
_RETRIEVE_URL = "http://localhost:{port}/retrieve_cards"

# Classe de aplicação
class AppInstance:
    def __init__(self, port: int):
        self.port = port
        self.logger = logging.getLogger("AppLogger")
        logging.basicConfig(level=logging.INFO)

# Função para associar cartão a telefone
def associate(app: AppInstance, cc: str, phone: str) -> bool:
    r = requests.post(
        url=_ASSOCIATE_URL.format(port=app.port),
        json={"credit_card": cc, "phone": phone},
    )
    if not r.ok:
        app.logger.warning(
            "POST /associate_card %s %s failed: %d %s",
            cc,
            phone,
            r.status_code,
            r.reason,
        )
        return False
    return True

# Função para recuperar cartões associados a telefones
def get_cc_assoc(app: AppInstance, phones: list[str]) -> list[str]:
    r = requests.post(
        url=_RETRIEVE_URL.format(port=app.port),
        json={"phone_numbers": phones}
    )
    if not r.ok:
        app.logger.warning(
            "POST /retreive_cards %s not found: %d %s", phones, r.status_code, r.reason
        )
        return []
    return sorted(r.json().get("card_numbers", []))

# Teste funcional da operação
def func_test_operation(app: AppInstance) -> bool:
    # Associações com cartões e telefones válidos
    if not associate(app, "4111111111111111", "+5511988887777"):  # Visa com DDI Brasil
        print("Falha ao associar Visa com DDI Brasil")
        return False
    if not associate(app, "4111111111111111", "+1(800)123-4567"):  # Mesmo cartão, outro número
        print("Falha ao associar Visa com outro número")
        return False
    if not associate(app, "5500000000000004", "+1(800)123-4567"):  # Mastercard, mesmo número
        print("Falha ao associar Mastercard com outro número")
        return False

    # Verificações de associação
    if get_cc_assoc(app, ["+5511988887777"]) != ["4111111111111111"]:
        print("Falha na verificação do primeiro número")
        return False
    if get_cc_assoc(app, ["+5511988887777", "+1(800)123-4567"]) != ["4111111111111111"]:
        print("Falha na verificação de múltiplos números")
        return False
    if get_cc_assoc(app, ["+5511988887777", "+1(800)123-4567", "+44 2079460958"]) != []:
        print("Falha na verificação de número não associado")
        return False
    if get_cc_assoc(app, ["+1(800)123-4567"]) != ["4111111111111111", "5500000000000004"]:
        print("Falha na verificação de múltiplos cartões associados a um número")
        return False

    return True

# Execução principal
if __name__ == "__main__":
    app = AppInstance(port=5000)  # Ajuste a porta conforme o serviço que você roda
    success = func_test_operation(app)
    print("Teste funcional:", "OK ✅" if success else "Falhou ❌")
