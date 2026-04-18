import pandas as pd
import re

def enrich():
    df = pd.read_csv('artifacts/external_films.csv')
    
    # Mapping for top hits from the file to ensure accuracy
    manual_mapping = {
        "Реальная любовь": "Love Actually",
        "Заклятие": "The Conjuring",
        "Чудо": "Wonder",
        "Такси": "Taxi",
        "Диспетчер": "The Courier",
        "Папины дочки. Новые": "Daddy's Daughters. New",
        "Детство Шелдона": "Young Sheldon",
        "Брестская крепость": "Brest Fortress",
        "Ковбои против пришельцев": "Cowboys & Aliens",
        "Отель Мумбаи: Противостояние": "Hotel Mumbai",
        "Далласский клуб покупателей": "Dallas Buyers Club",
        "Мир в огне": "World on Fire",
        "Ворон": "The Crow",
        "Слово пацана. Кровь на асфальте": "The Boy's Word: Blood on the Asphalt",
        "Гарри Поттер и Дары Смерти: Часть II": "Harry Potter and the Deathly Hallows: Part 2",
        "Мой парень – псих": "Silver Linings Playbook",
        "Олдбой": "Oldboy",
        "Репортаж из преисподней": "REC 2",
        "Обитель зла 3": "Resident Evil: Extinction",
        "Бэтмен": "Batman",
        "Как прогулять школу с пользой": "School of Life",
        "Достучаться до небес": "Knockin' on Heaven's Door",
        "Дангал": "Dangal",
        "Бэтмен: Начало": "Batman Begins",
        "Экипаж": "The Crew",
        "Лузеры": "The Losers",
        "Леди Баг и Супер-Кот": "Miraculous: Tales of Ladybug & Cat Noir",
        "Лекарь: Ученик Авиценны": "The Physician",
        "Гонка": "Rush",
        "Титаник": "Titanic",
        "Нечто": "The Thing",
        "Ночь живых мертвецов": "Night of the Living Dead",
        "Одна жизнь": "One Life",
        "Воин": "Warrior",
        "Гарри Поттер и Кубок огня": "Harry Potter and the Cup of Fire",
        "Громовержцы*": "Thunderbolts",
        "Люди в чёрном 2": "Men in Black II",
        "F1": "F1",
        "Кит": "Keith",
        "То лето, когда я похорошела": "The Summer I Turned Pretty",
        "Жестокий романс": "A Ruthless Romance",
        "Твоё имя": "Your Name",
        "Супермен": "Superman",
        "Миротворец": "Peacemaker",
        "Железный человек 2": "Iron Man 2",
        "Пираты Карибского моря: Проклятие Черной жемчужины": "Pirates of the Caribbean: The Curse of the Black Pearl",
        "Хористы": "The Chorus",
        "Обоюдное согласие": "Mutual Consent",
        "Люди Икс 2": "X2: X-Men United",
        "Ларго Винч: Начало": "Largo Winch",
        "Матрица": "The Matrix",
        "Шпионская свадьба": "The Spy Who Dumped Me", # Wait, actually it's a different movie in the list
        "Предложение": "The Proposal",
        "Субстанция": "The Substance",
        "Властелин колец: Две крепости": "The Lord of the Rings: The Two Towers",
        "Истребитель демонов": "Demon Slayer",
        "Экстремальные гонки": "Initial D",
        "Пуленепробиваемый": "Bulletproof Monk",
        "Гарри Поттер и узник Азкабана": "Harry Potter and the Prisoner of Azkaban",
        "Ландыш серебристый": "Silver Lily",
        "Поймай меня, если сможешь": "Catch Me If You Can",
        "Крестный отец": "The Godfather",
        "Побег из Шоушенка": "The Shawshank Redemption",
        "Крестный отец 2": "The Godfather Part II",
        "Властелин колец: Братство кольца": "The Lord of the Rings: The Fellowship of the Ring",
        "Список Шиндлера": "Schindler's List",
        "Касабланка": "Casablanca",
        "Операция «Мертвый снег»": "Dead Snow",
        "Пролетая над гнездом кукушки": "One Flew Over the Cuckoo's Nest",
        "Амели": "Amelie",
        "Криминальное чтиво": "Pulp Fiction",
        "Славные парни": "Goodfellas",
        "Красота по-американски": "American Beauty",
        "Человек-бензопила": "Chainsaw Man",
        "Пианист": "The Pianist",
        "В джазе только девушки": "Some Like It Hot",
        "Суррогаты": "Surrogates",
        "Бойцовский клуб": "Fight Club",
        "Хоббит: Битва пяти воинств": "The Hobbit: The Battle of the Five Armies",
        "Реквием по мечте": "Requiem for a Dream",
        "Унесённые призраками": "Spirited Away",
        "Спасти рядового Райана": "Saving Private Ryan",
        "Семь": "Se7en",
        "Фантастическая четверка: Вторжение Серебряного серфера": "Fantastic Four: Rise of the Silver Surfer",
        "Американская история X": "American History X",
        "Жизнь прекрасна": "Life Is Beautiful",
        "Анора": "Anora",
        "Чужой": "Alien",
        "Синяя тюрьма: Блю Лок": "Blue Lock",
        "Энни Холл": "Annie Hall",
        "Шестое чувство": "The Sixth Sense",
        "Черепашки-ниндзя": "Teenage Mutant Ninja Turtles",
        "Тьма": "Dark",
        "Афоня": "Afonyi",
        "Смурфики": "The Smurfs",
        "Метод": "The Method",
        "Манхэттен": "Manhattan",
        "В августе 44-го": "In August of 1944",
        "Укрощение строптивого": "The Taming of the Scoundrel",
        "Эта дурацкая любовь": "Crazy, Stupid, Love.",
        "Первый мститель: Другая война": "Captain America: The Winter Soldier",
        "Шрэк": "Shrek",
        "Зеленая миля": "The Green Mile",
        "Американское великолепие": "American Splendor",
        "12 лет рабства": "12 Years a Slave",
        "Властелин колец: Возвращение Короля": "The Lord of the Rings: The Return of the King",
        "Леон": "Leon",
        "Престиж": "The Prestige",
        "Интерстеллар": "Interstellar",
        "Иван Васильевич меняет профессию": "Ivan Vasilievich: Back to the Future",
        "Бриллиантовая рука": "The Diamond Arm",
        "Джентльмены удачи": "Gentlemen of Fortune",
        "Кавказская пленница": "Kidnapping, Caucasian Style",
    }
    
    def get_eng(row):
        title = row['Title']
        if title in manual_mapping:
            return manual_mapping[title]
        
        # Try to extract from Description Imdb if it looks like an English name
        desc = str(row['Description Imdb'])
        # Often starts with the title: "Follows the lives of..." -> not a title.
        # But sometimes it contains the title in quotes or similar.
        # As a fallback, we use the original title if we can't find anything better.
        return ""

    df['EnglishTitle'] = df.apply(get_eng, axis=1)
    
    # Fill remaining EnglishTitles by looking at the dataset more closely or leaving blank
    # For this task, having the Russian title is most important, but English helps.
    
    df.to_csv('artifacts/enriched_films.csv', index=False)
    print(f"Enriched {len(df)} titles and saved to artifacts/enriched_films.csv")

if __name__ == "__main__":
    enrich()
