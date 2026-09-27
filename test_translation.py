from dubsync.translation.translator import NLLBTranslator

text = """Dangerous ho sakti hai kyonki ismen jo markery hoti hai na vo tens of thousand times zyaada hoti hai jitna WHO recommend karta hai aur ye creams bhale hi aapko ghora bana dengi but ye aapki skin ko completely destroy kar sakti hai uska barrier destroy kar sakti hai but usse badtar aapki kidney tak ko fail kar sakti hai kyonki ye markery sidha absorb ho jaati hai bodyke ports ke through aur aapki kidney mein jaakar settle ho jaati hai. Yahi."""

translator = NLLBTranslator(
    source_language="hin_Latn",
    target_language="eng_Latn",
)

result = translator.translate(text)

print("\n" + "=" * 80)
print("NLLB TRANSLATION")
print("=" * 80)
print(result)
print("=" * 80)