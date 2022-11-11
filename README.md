1. PROJEKT
Projekt to prosta implementacja rozpoznawania mowy z plików audio
Pliki audio stanowią wersy wiersza Wisławy Szymborskiej, użytego w piosence Sanah - "Nic dwa razy"
Pliki pobierane są z folderu "sanah"

2. Przygotowanie środowiska:

Za pomocą komendy pip install, zainstaluj poniższe biblioteki wpisując je w terminalu:

- biblioteka Transformers instalowana z Githuba
pip install https://github.com/huggingface/transformers/archive/refs/heads/master.zip

- biblioteka do dekodowania modelem języka
pip install https://github.com/kensho-technologies/pyctcdecode/archive/refs/heads/main.zip - 

- biblioteka do obsługi modelu języka w dekoderze
pip install https://github.com/kpu/kenlm/archive/master.zip - 

- biblioteka do wczytywania plików audio
pip install wavio

- biblioteka do liczenia WER 
pip install jiwer 

- mała biblioteka do wczytywania i oglądania modeli języka
pip install arpa

3. Uruchomienie:

- przenawiguj terminal na ścieżkę z projektem
- uruchom program komendą "python3 main.py"

4. REZULTAT:
>>sanah6
najtępszymi w szkolę świata
najtępszymi w szkole świata

>>sanah19
róża jak wygląda ruża
róża jak wygląda róża

>>sanah15
tak ni byo jakróżo
tak mi było jakby róża

>>sanah9
żaden dzień się nie powtórzy
żaden dzień się nie powtórzy

>>sanah5
choć byśmy uczniami byli
choćbyśmy uczniami byli

>>sanah8
żadnejzimy ani lata
żadnej zimy ani lata

>>sanah4
i pomożemy bez rutyny
i pomrzemy bez rutyny

>>sanah3
zrodziliśmy się bez wprawy
zrodziliśmy się bez wprawy

>>sanah13
wczoraj kiedy trwoje imie
wczoraj kiedy twoje imię

>>sanah18
odwróciłam trwa szkuścianie
odwróciłam twarz ku ścianie

>>sanah7
nie będziemy repetować
nie będziemy repetować

>>sanah11
dwóch tych samych posawunków
dwóch tych samych pocałunków

>>sanah20
czy to kwiat a norzekami
czy to kwiat a może kamień

>>sanah2
i nie zdarzy ztej przyczyny
i nie zdarzy z tej przyczyny

>>sanah12
dwóch jednakich spojrzeńw ocze
dwóch jednakich spojrzeń w oczy

>>sanah10
nie ma dwóch podobnych nocy
nie ma dwóch podobnych nocy

>>sanah14
ktoś wymówił przymnie głośno
ktoś wymówił przy mnie głośno

>>sanah16
przez otwarte w padła okno
przez otwarte wpadła okno

>>sanah17
dziś kiedy jesteśmy razem
dziś kiedy jesteśmy razem

>>sanah1
nic twa razy się nie zdarza
nic dwa razy się nie zdarza

{'wer': 0.3258426966292135, 'mer': 0.31868131868131866, 'wil': 0.48582129481005876, 'wip': 0.5141787051899412, 'hits': 62, 'substitutions': 20, 'deletions': 7, 'insertions': 2}



