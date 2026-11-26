"""Sort articles into a few broad families by what kind of thing they are, for coloring the map.

    uv run python -m articular.pipeline.build_kind_families --wiki enwiki

Reads data/<wiki>_kind_graph.parquet (articular/pipeline/kind_graph.py). Each article's "is a kind of" chain is
followed upward until it meets one of the articles listed in FAMILIES below; that decides the family.
The lists were made by hand from the commonest kinds in English Wikipedia. A kind is filed by what the
articles under it are about, which is not always what its own article is about (see People).

Articles whose opening sentence names no kind are then filed, where possible, by Wikidata: the classes
their Wikidata item is an instance of are followed up the subclass tree to the nearest class that is the
item of a listed article (or one of the items in WIKIDATA_ITEMS). This needs data/wd/ (scripts/fetch_wikidata.sh)
and is skipped without it. Titles that make the kind plain are used too: the sure patterns before Wikidata, the looser ones after (TITLES_FIRST, TITLES_LAST).

Writes data/<wiki>_kind_families.parquet (page_id, family) and data/<wiki>_kind_families.json (the
family names, in order). The last two families are "Other kinds" (a kind was found but its chain meets
no listed article) and "No kind found".
"""

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from articular.experiments.score_rules import read_items

FAMILIES = {
    # Many of these are activities, because a job title often redirects to one ("singer" -> Singing,
    # "cricketer" -> Cricket), and nearly every article filed under such a kind is about a person.
    "People": """Person; Human; Politician; Actor; Writer; Artist; Lawyer; Officer; Football player; Journalist;
        Businessperson; Author; Poet; Professor; Composer; Champion; Musician; Historian; Architect; Teacher;
        Medical doctor; Athlete; Engineer; Pitcher; Diplomat; Scholar; Songwriter; Prelate; Scientist; General officer;
        Priest; Soldier; Novelist; Economist; Physicist; Bishop; Mathematician; Photographer; Judge; Official; Pianist;
        Designer; Clergy; Chemist; Nobility; Jurist; Model; Midfielder; Film director; Manager (association football);
        Playwright; Merchant; King; Farmer; Domestic worker; Professional golfer; Defenceman; Comedian; Son; Guitarist;
        Screenwriter; Woman; Defender (association football); Outfielder; Psychologist; Commander; Aircraft pilot;
        United States federal judge; Surgeon; Biologist; Sailor; Astronomer; Cartoonist; Ruler; Women in business;
        Prince; Barrister; Rabbi; Man; Goalkeeper (association football); Geologist; Cinematographer; Business magnate;
        Soprano; Illustrator; People; Drummer; Catcher; Goaltender; Linebacker; Disc jockey; Critic; Quarterback; Chef;
        Princess; Coach (sport); Wife; Daughter; Computer scientist; Missionary; Serial killer; Infielder; Emeritus;
        Anthropologist; Expert; Health professional; Individual; Personality;
        Coach (basketball); Police officer; Chief executive officer; Theatre director; Sports commentator;
        Member of parliament; Tenor; Coach (baseball); Advocate; Admiral; Geographer; Consultant; Television
        director; Gangster; Ambassador; Baritone; Agriculturist; Audio engineer; Lord; YouTuber; Head coach;
        Art director; Graphic designer; Celebrity; Forward (association football); Cornerback; Centre (ice
        hockey); Defensive end; Guard (gridiron football); Defensive back; Defensive tackle; First baseman;
        Shortstop; Second baseman; Third baseman; Winger (ice hockey); Film producer; Model (person);
        Noblewoman; Jockey; Lineman (gridiron football); Cardinal (Catholic Church); Tackle (gridiron football
        position); Baronet; Goalkeeper; Flying ace; Secretary; Safety (gridiron football position); Accountant;
        Child; Lieutenant general; Tight end; Lecturer; Dentist; Commissioner; Friar; Samurai; Bishops in the
        Catholic Church; List of entertainer occupations; United States House of Representatives;
        Singing; Rapping; Activism; Filmmaking; Entrepreneurship; Conducting; Voice acting; Film editing;
        Choreography; Philanthropy; Editing; Statecraft; Leadership; Management; Nursing; Voice acting in Japan;
        Fashion design; Sports journalism; Photojournalism; Stand-up comedy; Bookselling; Translation; Crime; Dance;
        Cricket; Competitive swimming; Cycle sport; Rowing (sport); Boxing; Sprint (running); Shooting sports;
        Running; Cycling; Long-distance running; Road bicycle racing; Judo; Professional wrestling; Professional
        boxing; Gymnastics; Canoe sprint; Alpine skiing; Sailing (sport); Ice hockey; Sport of athletics; Wrestling;
        Amateur wrestling; Olympic weightlifting; Cross-country skiing (sport); Fencing; Mixed martial arts; Curling;
        Hurling; Figure skating; Middle-distance running; Bobsleigh; Speed skating; Track and field; Equestrianism;
        Artistic gymnastics; Biathlon; Rhythmic gymnastics; Archery; Skiing; Hurdling; Race walking; Freestyle
        swimming; Motorcycle racing; Squash (sport); Ice dance; Freestyle skiing; Modern pentathlon; Pair skating;
        High jump; Mountaineering; Kickboxing; Long jump; Amateur boxing; Para swimming; Weightlifting; Short-track
        speed skating; Pole vault; Karate; Triathlon; Triple jump; Shot put; Australian rules football; Gaelic
        football; American football""",
    "Settlements and regions": """Village; Town; Communes of France; Classification of inhabited localities in
        Russia; Unincorporated area; Municipality; City; Neighbourhood; District; Census-designated place; Area;
        Electoral district; Community; Suburb; Comune; Historic districts in the United States; Civil parish; Civil
        township; Hamlet; Human settlement; Municipalities of Brazil; Villages of Turkey; Township; Region;
        Hamlet (place); Suburbs and localities (Australia); Village development committee (Nepal);
        Municipalities of Germany; County (United States); Municipalities of Switzerland; Electoral district (Canada);
        Administrative divisions of New York (state); Communes of Romania; Municipalities of Poland; Ghost town;
        Urban areas in Sweden; Census town; County; Municipalities of Spain; Parish; Municipalities of the
        Philippines; Diocese; Location; Place (United States Census Bureau); Country; School district;
        Residential area; Districts of Peru; Town (China); Township (Pennsylvania); Townland; Rural municipality;
        Rural settlement; New England town; Borough (Pennsylvania); Subdistrict (China); Types of rural
        communities; Barrios of Puerto Rico; Village (United States); Historic district; Districts of Ethiopia;
        Union councils of Bangladesh; Union councils of Pakistan; Community development block; Indian reserve;
        Corregimientos of Panama; United Kingdom constituencies; Capital city; Polity;
        Freguesia; Local government in Australia; Populated places in Ukraine; Constituencies of the Parliament of
        the United Kingdom; Administrative divisions of Wisconsin; Settlement (Croatia); Cantons of France; Market
        town; List of constituencies of the National Assembly of France; District (Vietnam); Tambon; Dispersed
        settlement; Districts of Algeria; Districts of Costa Rica; Municipalities of Belgium; Commune (Vietnam);
        Ranchos of California; Ward (Vietnam); Township (China); Districts of Turkey; County (China)""",
    "Natural features and space": """River; Mountain; Lake; Island; Tributary; Stream; Geological formation;
        Summit; Hill; Asteroid; Star; Reservoir; Bay; Impact crater; Valley; Mountain range; Body of water; Landform;
        Waterfall; Mountain pass; Cave; Glacier; Coast; Binary star; Spiral galaxy; Park; National park; Recreation
        area; Site of Special Scientific Interest; Volcano; Stratovolcano; Exoplanet; Fjord; Canyon; Petroleum
        reservoir""",
    "Living things": """Moth; Beetle; Sea snail; Flowering plant; Plant; Species; Butterfly;
        Actinopterygii; Ground beetle; Bird; Family (taxonomy); Bacteria; Fungus; Frog; Orchid; Fly; Genus; Land snail;
        Spider; Pyralidae; Tree; Lizard; Bee; Shrub; Sea slug; Rodent; Longhorn beetle; Leaf beetle; Fish; Ant;
        Jumping spider; Organism; Phylum; Order (taxonomy); Taxonomic rank; Clade; Vertebrate; Colony (biology);
        Animal; Mammal; Insect;
        Thoroughbred; Mite; Hoverfly; Dinosaur; Grasshopper; Weevil; Catfish; Fern; Damselfly; Cicada;
        Nematode; Scarabaeidae; Green algae; Big cat; Felidae; Eukaryote;
        Subspecies; Linyphiidae; Cichlid; Tiger beetle; Wasp; Cultivar; Tephritidae; Cyperaceae; Herb; Miridae;
        Proteaceae; Salamander; Gastropod; Bivalvia; Skipper (butterfly); Cactus; Conifer; Endemism; Predation;
        Parasitism; Family""",
    "Creative works": """Album; Film; Song; Painting; Drama (film and television); Novel; Book; Comedy film; Episode;
        Newspaper; Game; Sculpture; Documentary film; Extended play; Magazine; Academic journal; Action film;
        Video game; Western film; Crime film; Thriller film; Compilation album; Horror film; Romance film; Opera;
        Poetry; Drama; Sitcom; Adventure film; Television show; Short film; Manuscript; Telenovela;
        Silent film; Television film; Comedy drama; Musical film; Anthology; Musical composition;
        Film genre; Publication; Periodical literature; Short story;
        Romantic comedy; Reality television; Television documentary; Animated series; Game show; Soap opera;
        Adventure game; Expansion pack; Play (theatre); Platformer; Web series; Miniseries; Memoir; Essay;
        Historical drama; Superhero; Supervillain; Children's literature; Crime fiction; Fantasy literature;
        Detective fiction; Limited series (comics); Single (music); Cartoon; Serial film; Action role-playing game;
        Shoot 'em up; Media franchise; Autobiography; Chronicle; Folk music; Music; Art; Visual arts; Music genre;
        Literature; The arts; Performing arts; DVD; Adaptation; Genre; Television; Performance""",
    "Organizations": """Company; Team; Organization; School; Radio broadcasting; Secondary school; Party; Institution;
        Bank; Political party; Regiment; University; Trade union; Nonprofit organization; College; Airline; Institute;
        Record label; Society; Television station; State school; Educational institution; Business;
        Financial institution; Musical ensemble; Sports club; Manufacturing; Publishing;
        Football team; Research institute; Television channel; Governing body; Ministry (government
        department); Private university; Military organization; Professional association; Division (military);
        Day school; Chain store; Production company; Non-governmental organization; Committee; Advocacy group;
        Legislature; Parliament; Battalion; Diplomatic mission; Council; Brigade; Coalition; Community
        college; Television broadcaster; Cycling team; State legislature; Athletic conference; International
        organization; Architectural firm; Girl group; Corps; Trade union federation; Federation; Squadron
        (aviation); Christian school; Political movement;
        Orchestra; Retail; Studio; Alliance; Consortium; Conglomerate (company); College fraternities and
        sororities; Artillery battery; Deliberative assembly; Voluntary association; Senate; Chain; Force;
        Broadcasting; Low-power broadcasting; Collective""",
    "Sports and competitions": """Association football; Horse racing; Auto racing; First-class cricket; Women's
        association football; Physical activity; Sport; Combat sport; Motorsport; Team sport; Tournament;
        Competition; Season; Championship; Match; Stock car racing; Bowl game; Season (sports); Paralympic sports;
        Professional wrestling championship; Gridiron football; College football; Multi-sport event; Road racing;
        Football in Germany; Cup; Play (activity)""",
    "Events and history": """Festival; Battle; Conference; Engagement; Film festival; Concert tour; Rebellion; War;
        Military operation; Awards ceremony; Engagement (military); Meeting; Tropical cyclone; Storm; Ceremony;
        Exhibition; Siege; Election; By-election; General election; Referendum; Earthquake; Massacre; Riot; Protest;
        Flight; Century; Disaster; Incident Command System""",
    "Buildings and structures": """Building; House; Train station; Airport; Museum; Stadium; Home; Bridge; Castle;
        Metro station; Hospital; Hotel; Restaurant; Dam; Parish church; Arena; Theatre; Church (building);
        Fortification; Temple; Monastery; Skyscraper; Listed building; Mosque; Monument; Residence;
        Aerodrome; Multi-purpose stadium; Palace; Lighthouse; Library; Shopping mall; Prison; Office; Facility;
        Structure; Historic house; Tram stop; Shopping center; Manor house; Power station; Tunnel; Chapel;
        International airport; Synagogue; Memorial; Commercial property; Headquarters; Art gallery;
        State highway; Road; Highway; Street; Thoroughfare; Cemetery; Archaeological site; Town square; Square;
        Hindu temple; Tower; Hall; Villa; Ballpark; Rail transport; Trail; Regional road (Ireland); National
        highways of Japan; List of non-building structure types; Landmark; Garden; Campus; Resort""",
    "Vehicles, machines and technology": """Ship; Car; Aircraft; Train; Tool; Invention; Musical instrument; Motor
        vehicle; Physical object; Weapon; Gun; Smartphone; Sailboat; Cargo ship; Submarine; Steamship; Ferry; Lead
        ship; Cello; Satellite; Ship class; Schooner; Canoe; Type VII submarine; String instrument; Software;
        Website; Engine; Database; Vehicle; Biplane; Monoplane; Machine; Communications satellite; Computer; Boat;
        Yacht; Steamboat; Violin; Technology""",
    "Chemistry, biology and medicine": """Protein; Enzyme; Drug; Organic compound; Oil; Biomolecule; Chemical
        compound; Inorganic compound; Mineral; Molecule; Disease; Plant disease; Gene; Material; Mixture; State of
        matter""",
    "Fields and ideas": """Philosophy; System; Process; Project; Research; Archaeology; Linguistics; Sociology;
        Theology; Concept; Theory; Statistic; Botany; Entomology; Property; Result; Justice; Academic discipline;
        Political science; History; Classics; Literary criticism; Musicology; Natural history; Zoology; Philology;
        Astrophysics; Ornithology; Paleontology; Neurology; Time; Scientific theory; Explanation; Phenomenon; Number;
        Electrical engineering; Pediatrics; Pathology; Physiology; Theoretical physics; Epidemiology; Surveying;
        Ecology; Ophthalmology; Immunology; Algorithm; Sequence; Space; Motion; Thermodynamic system; Ability;
        Skill; Classification; Entity; Education; Feminism""",
    "Language and names": """Surname; Name; Title; Language; Given name; Italian name; Brand; Label; Terminology;
        Word; Language family; Abbreviation; German name; English name; French name; Polish name; Common name;
        Stage name; Nickname; Finnish name; Pseudonym; Patronymic surname; Czech name; Turkish name; Dutch
        name; East Slavic name; Spanish naming customs; Scandinavian family name etymology; List of common
        Japanese surnames; Masculinity; Phrase; Symbol""",
    "Society, culture and religion": """Law; Act of Parliament (United Kingdom); Land tenure; Ownership; Award;
        Prize; Honour; Literary award; Treaty; Citizenship; Ethnicity; Currency; Tribe; Aristocracy (class); Clan;
        Dynasty; Goddess; God; Deity; Religion; Drink; Food; Suffrage; Immigration; Civil service; Medal; Rank;
        Social group; Bilateralism""",
    "Lists and timelines": """List; Timeline; Year; Lists of musicians; Discography; Day; Record chart; Chart""",
}
# For the Wikidata step. A family's classes are the Wikidata items of its listed articles plus these ones,
# given by article title or, where English Wikipedia has no article for the class, by item number.
# People uses only what is listed here: its FAMILIES entries are mostly activities, not classes of person.
WIKIDATA = {
    "People": "Human; Person",
    "Settlements and regions": "Administrative division; Geographic region; State (polity)",
    "Natural features and space": """Astronomical object; Protected area; Galaxy; Planet; Minor planet; Nature
        reserve; Forest; Beach; Peninsula; Cape (geography); Desert; Wetland; Nebula; Star cluster""",
    "Living things": "Taxon; Breed",
    "Creative works": """Creative work; Work of art; Literary work; Television program; Film series; Book series;
        Character (arts); Comics; Radio program; Podcast; Comic book; Manga; Anime""",
    "Organizations": "Sports team; Military unit; Government agency; Religious organization; Political organisation",
    "Sports and competitions": "Sports league; Sports competition",
    "Events and history": "Natural disaster; Aviation accidents and incidents; Coup d'état; Trial; Protest",
    "Buildings and structures": "Infrastructure; Railway line; Canal; Port; Mine (mining)",
    "Vehicles, machines and technology": "Spacecraft; Locomotive; Rocket; Product (business)",
    "Chemistry, biology and medicine": "Chemical substance; Chemical element; Medication; Protein family; Anatomy",
    "Fields and ideas": "Science; Branches of science; Theorem; Mathematical object; Ideology",
    "Language and names": "Writing system; Letter (alphabet); Dialect",
    "Society, culture and religion": "Ethnic group; Dish (food); Public holiday; Profession; Mythology",
}
# Listed articles too general to stand for a family in Wikidata's class tree, where almost anything sits under them
WIKIDATA_SKIP = """Entity; Concept; Phenomenon; System; Result; Property; Process; Physical object; Location;
    Structure; Facility; Infrastructure; Genre; Product (business); Publication; Project; Research; Time; Statistic;
    Space; Motion; Ability; Skill; Classification; Sequence; Area; Community; Title; Name; Label; Brand; Home; Office;
    Material; Mixture; Society; Party; Institution; Match; Engagement; Meeting; Season; Day; Tool; Invention;
    Technology; Explanation; Justice; Ownership; Art; The arts; Performance; Television; Music; Literature;
    Physical activity; Play (activity); Competition"""
WIKIDATA_SKIP = {" ".join(t.split()) for t in WIKIDATA_SKIP.split(";")}
WIKIDATA_STOP = (4167410, 15623926)      # disambiguation pages and set index articles: not a kind of thing
WIKIDATA_ITEMS = {      # item number: family
    5: "People", 16521: "Living things", 13406463: "Lists and timelines", 1656682: "Events and history",
    43229: "Organizations", 17537576: "Creative works", 811979: "Buildings and structures",
    56061: "Settlements and regions", 486972: "Settlements and regions", 6999: "Natural features and space",
    27020041: "Sports and competitions", 16510064: "Sports and competitions", 13406554: "Sports and competitions",
}


def by_wikidata(args, graph, none, where):
    """Family for each article in `none` (positions in graph) that Wikidata can place, as (positions, families)."""
    statements, props = args.data / "wd" / "statements.tsv.gz", args.data / "wd" / f"{args.wiki}-page_props.sql.gz"
    if not (statements.exists() and props.exists()):
        print("no Wikidata files under data/wd: skipping the Wikidata step")
        return np.array([], dtype=int), np.array([], dtype=int)
    con = duckdb.connect()
    con.sql(f"set temp_directory = '{args.data}/duckdb_tmp'; set preserve_insertion_order = false;")
    con.register("items_raw", read_items(props))
    con.register("todo_raw", pd.DataFrame({"pos": none, "page_id": graph["page_id"].to_numpy()[none]}))
    con.sql("create table todo as select t.pos, i.q from todo_raw t join items_raw i using (page_id)")

    rows = [(q, NAMES.index(name), f"Q{q}") for q, name in WIKIDATA_ITEMS.items()] + [(q, NONE, f"Q{q}") for q in WIKIDATA_STOP]
    listed = {name: (WIKIDATA[name] if name == "People" else FAMILIES[name] + ";" + WIKIDATA.get(name, "")) for name in FAMILIES}
    titles = pd.DataFrame([(" ".join(t.split()), NAMES.index(name)) for name, text in listed.items() for t in text.split(";")
                           if t.strip() and " ".join(t.split()) not in WIKIDATA_SKIP],
                          columns=["title", "family"])
    titles = titles[titles["title"].isin(where.index)].copy()
    titles["page_id"] = graph["page_id"].to_numpy()[where.reindex(titles["title"]).to_numpy()]
    con.register("titles", titles)
    con.register("fixed", pd.DataFrame(rows, columns=["q", "family", "title"]))
    con.sql("""create table anchors as
               select q, any_value(family) as family, any_value(title) as title from (
                   select q, family, title from fixed
                   union all select i.q, t.family, t.title from titles t join items_raw i using (page_id)
                   where i.q not in (select q from fixed)) group by q""")

    con.sql(f"""
    create table st as
    select item, prop, value from (
        select cast(substr(column0, 2) as bigint) as item, cast(substr(column1, 2) as int) as prop,
               cast(substr(column2, 2) as bigint) as value
        from read_csv('{statements}', delim = '\t', header = false, quote = '', escape = '', auto_detect = false,
                      columns = {{'column0': 'varchar', 'column1': 'varchar', 'column2': 'varchar'}})
    ) where prop = 279 or (prop = 31 and item in (select q from todo));
    create table up as select distinct value as start, value as anc, 0 as depth
    from st where prop in (31, 279) and item in (select q from todo);
    """)
    for depth in range(1, 12):
        before = con.sql("select count(*) from up").fetchall()[0][0]
        con.sql(f"""insert into up select distinct u.start, e.value, {depth}
                    from up u join st e on e.item = u.anc and e.prop = 279
                    where u.depth = {depth - 1} and not exists (select 1 from up x where x.start = u.start and x.anc = e.value)""")
        if con.sql("select count(*) from up").fetchall()[0][0] == before:
            break
    # the nearest listed class wins; of two equally near, the lower family number
    con.sql("""
    create table placed as
    select t.pos, arg_min(a.family, u.depth * 100 + a.family) as family, arg_min(a.title, u.depth * 100 + a.family) as via
    from todo t join st s on s.item = t.q and s.prop in (31, 279)
    join up u on u.start = s.value join anchors a on a.q = u.anc
    group by t.pos
    """)
    print(f"Wikidata: {len(none):,} articles with no kind, {con.sql('select count(*) from todo').fetchall()[0][0]:,} have an item, "
          f"{con.sql(f'select count(*) from placed where family <> {NONE}').fetchall()[0][0]:,} placed")
    con.register("names", pd.DataFrame({"pos": none, "title": graph["title"].to_numpy()[none]}))
    for family, via, count, examples in con.sql("""
            select p.family, p.via, count(*) as c, list(n.title order by hash(n.title))[:4]
            from placed p join names n using (pos) group by all order by c desc limit 60""").fetchall():
        print(f"  {count:>8,}  {NAMES[family]} via {via}: {'; '.join(examples)}")
    out = con.sql(f"select pos, family from placed where family <> {NONE}").df()
    return out["pos"].to_numpy(), out["family"].to_numpy()


# Articles whose opening sentence names no kind are filed by their title where the title makes it plain
SPORT_WORDS = (r"season|team|football|soccer|rugby|cricket|tennis|basketball|baseball|hockey|golf|cycling|athletics|"
               r"championship|cup|grand prix|open|tour|league|olympics|games|tournament|race|rally|series|masters|"
               r"trophy|classic|playoffs|draft|bowl|500|singles|doubles|relay|metres|marathon")
TITLES_FIRST = [
    ("Lists and timelines", r"^(?:lists? of|outline of|index of|glossary of|timeline of|bibliography of) "
                            r"| (?:discography|filmography|bibliography|videography)$|^[0-9]{1,4}(?: bc| ad)?$"
                            r"|^(?:january|february|march|april|may|june|july|august|september|october|november|december) [0-9]{1,2}$"
                            r"|^[0-9]{3,4}s?(?: bc)? in |^(?:ad )?[0-9]{1,4}s?(?: bc)?$"),
    ("Sports and competitions", r" at the [0-9]{4} "),
]
TITLES_LAST = [
    ("Sports and competitions", rf"^[0-9]{{4}}(?:[–-][0-9]{{2,4}})? .*\b(?:{SPORT_WORDS})\b"),
    ("Events and history", r"^[0-9]{3,4}(?:[–-][0-9]{2,4})? "),
]
NAMES = list(FAMILIES) + ["Other kinds", "No kind found"]
OTHER, NONE = len(FAMILIES), len(FAMILIES) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wiki", default="enwiki")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--top", type=int, default=40, help="how many unsorted kinds to print")
    args = parser.parse_args()

    graph = pd.read_parquet(args.data / f"{args.wiki}_kind_graph.parquet", columns=["page_id", "title", "kind_page_id"])
    n = len(graph)
    index = pd.Series(np.arange(n), index=graph["page_id"].to_numpy())
    kind = np.where(graph["kind_page_id"] >= 0, index.reindex(graph["kind_page_id"]).fillna(-1).to_numpy(), -1).astype(int)

    anchor = np.full(n, -1)
    where = pd.Series(np.arange(n), index=graph["title"].to_numpy())
    for number, titles in enumerate(FAMILIES.values()):
        titles = [" ".join(t.split()) for t in titles.split(";")]
        missing = [t for t in titles if t not in where.index]
        if missing:
            print(f"{NAMES[number]}: no such article: {', '.join(missing)}")
        anchor[where.reindex([t for t in titles if t in where.index]).to_numpy()] = number

    # walk up: an article takes the family of the first listed article on its chain, not counting itself
    family, at = np.full(n, -1), kind.copy()
    for _ in range(args.max_steps):
        live = (family < 0) & (at >= 0)
        hit = live & (anchor[np.clip(at, 0, None)] >= 0)
        family[hit] = anchor[at[hit]]
        live &= ~hit
        at[live] = kind[at[live]]
    family[family < 0] = np.where(kind[family < 0] >= 0, OTHER, NONE)
    lower = graph["title"].str.lower()

    def by_title(patterns):                      # earlier patterns win
        for name, pattern in patterns:
            match = (family == NONE) & lower.str.contains(pattern, regex=True).to_numpy()
            family[match] = NAMES.index(name)
            print(f"filed by title under {name}: {match.sum():,}")

    by_title(TITLES_FIRST)
    pos, fam = by_wikidata(args, graph, np.flatnonzero(family == NONE), where)
    family[pos] = fam
    by_title(TITLES_LAST)

    pd.DataFrame({"page_id": graph["page_id"], "family": family.astype(np.int8)}).to_parquet(
        args.data / f"{args.wiki}_kind_families.parquet", index=False)
    (args.data / f"{args.wiki}_kind_families.json").write_text(json.dumps(NAMES))
    counts = np.bincount(family, minlength=len(NAMES))
    print(pd.DataFrame({"family": NAMES, "articles": counts, "share": (100 * counts / n).round(1)}).to_string(index=False))
    other = pd.Series(graph["title"].to_numpy()[kind[family == OTHER]]).value_counts().head(args.top)
    print("\ncommonest kinds left in 'Other kinds':")
    print("; ".join(f"{t} {c:,}" for t, c in other.items()))


if __name__ == "__main__":
    main()
