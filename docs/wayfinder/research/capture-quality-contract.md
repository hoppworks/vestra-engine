# Aufnahme- und Qualitätsvertrag: 3D-Floorplan aus RGB-Video

## Entscheidung

Der MVP akzeptiert ein einzelnes normales Smartphone-RGB-Video und **einen
manuell gesetzten Maßstabsanker**. Er liefert einen Floorplan nur, wenn sowohl
die Aufnahme als auch die Rekonstruktion die untenstehenden Gates passieren.
Andernfalls muss er mit konkreten, räumlich markierten Gründen zur Neuaufnahme
auffordern; er darf keine scheinbar präzisen Maße aus einer schlechten
Rekonstruktion ausgeben.

Ein Anker bestimmt einen globalen Maßstab, beweist aber nicht unabhängig die
Maßhaltigkeit an anderer Stelle. Deshalb lautet der Status ohne zweite Messung
stets **„maßstabskaliert, nicht extern verifiziert“**. Ein optionaler,
unabhängiger Kontrollanker macht den Status **„metrisch verifiziert“** möglich.
Diese Unterscheidung ist eine zwingende Produktentscheidung, keine UX-Option.

## Befundlage

- COLMAP behandelt ein Video als sequenzielle Bildmenge: *Sequential Matching*
  ist ausdrücklich für Bilder in zeitlicher Reihenfolge vorgesehen, setzt
  visuellen Überlapp voraus und kann Schleifen über einen Vocabulary Tree
  erkennen. Seine geometrisch geprüften Inlier sind die für die Rekonstruktion
  verwendeten Korrespondenzen. [COLMAP Tutorial](https://colmap.github.io/tutorial.html)
- Ein Kamera-Modell umfasst Intrinsics und Verzerrung; die einzelnen Bilder
  haben dagegen eigene Extrinsics. Ein Video darf daher während einer Aufnahme
  weder Zoom noch Objektiv/Kamera wechseln. [COLMAP Datenbankformat](https://colmap.github.io/legacy/3.10/database.html)
- ReCap empfiehlt mindestens drei Perspektiven je Szene, mindestens 60 %
  Überlapp aufeinanderfolgender Bilder und mindestens 20 % quer dazu. Für den
  MVP ist das eine Untergrenze, nicht das Ziel. [Autodesk ReCap – Scene Geometry
  and Materials](https://help.autodesk.com/cloudhelp/2018/ENU/Reality-Capture/files/GUID-CC48BA64-0CDA-42C6-AC78-518587672CE3.htm)
- OpenCV definiert den Reprojektionfehler als Abstand zwischen beobachteten und
  aus Kamera-/Pose-/3D-Punkt-Parametern zurückprojizierten Bildpunkten; kleiner
  ist besser. COLMAP speichert den Fehler seiner 3D-Punkte in Pixeln nach
  globalem Bundle Adjustment. Das macht Pixel-Reprojektion zu einem sinnvollen
  internen Gate, aber nicht zu einem Meter-Genauigkeitsnachweis.
  [OpenCV-Kalibrierung](https://docs.opencv.org/5.0/py_tutorials/py_calib3d/py_calibration/py_calibration.html),
  [COLMAP-Modellformat](https://colmap.readthedocs.io/en/latest/format.html)
- Für Trajektorien sind Absolute Trajectory Error (globale Konsistenz) und
  Relative Pose Error (lokaler Drift) etablierte, getrennte Metriken. Der
  TUM-RGB-D-Benchmark beschreibt ATE als SLAM- und RPE als
  Odometry-/Drift-Metrik. Ohne äußere Ground Truth können sie im Kundenlauf
  nicht absolut berechnet werden, sollen aber im Testkorpus benutzt werden.
  [TUM RGB-D Evaluation Tools](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools)

Die konkreten Zahlen weiter unten sind daher bewusst konservative
**MVP-Produktgrenzen**. Sie sind nicht als universelle wissenschaftliche
Schwellenwerte ausgegeben und müssen mit einem gelabelten Wohnungs-Testkorpus
nachkalibriert werden.

## Aufnahmevertrag

### Muss vor der Aufnahme sichtbar sein

1. **Eine zusammenhängende Rundtour.** Mit derselben Kamera im Uhrzeigersinn
   durch alle Räume gehen und wieder zu einem bereits eindeutig sichtbaren
   Startbereich zurückkehren. Jeder Raumübergang (insbesondere Türen, Flure
   und Ecken) muss langsam aus beiden Richtungen gezeigt werden. Daraus entstehen
   die Verbindungskanten und mindestens eine beobachtbare Schleife.
2. **Wände aus mehreren Blickrichtungen.** Jede später exportierte Wand muss
   in mindestens drei deutlich verschiedenen Ansichten vorkommen. 70–85 %
   Bildüberlapp zwischen benachbarten Keyframes ist das Ziel; 60 % ist die
   absolute Untergrenze aus der zitierten Photogrammetrie-Leitlinie.
3. **Langsam, gleichmäßig, mit Parallaxe.** Mit etwa 0,3–0,8 m/s gehen, nicht
   am selben Punkt drehen. Die Kamera meist auf 1,2–1,6 m Höhe, annähernd
   waagerecht bis höchstens ca. 20° nach unten halten, damit Wand-Boden-Kanten,
   Ecken und Textur gleichzeitig sichtbar sind. Pro Raum einmal den Rand
   abgehen und bei großen/offenen Räumen eine zweite, versetzte Bahn aufnehmen.
4. **Konstante Optik und Licht.** 1×-Hauptkamera, keine Zoom-, Objektiv-,
   Hoch-/Querformat- oder FPS-Wechsel innerhalb des Videos. Keine Spiegel,
   sich bewegenden Personen, Fernseher oder offene Fenster als dominierende
   Bildfläche. Gleichmäßiges Licht; nicht gegen helle Fenster filmen.
5. **Maßstabsanker.** Eine starre, gerade Strecke von 0,5–5 m mit einem
   Maßband messen (z. B. zwei klare Eckpunkte einer Wand) und ihre beiden
   Endpunkte während mindestens drei Ansichten sichtbar halten. Die Länge wird
   nach dem Upload eingegeben und beide Endpunkte im Video markiert.
6. **Optionaler Kontrollanker.** Eine zweite, räumlich getrennte gemessene
   Strecke (vorzugsweise in einem anderen Raum) aufnehmen. Sie darf nicht zur
   Skalierung verwendet werden; sie prüft ausschließlich die Ausgabe.

### Technische Eingangsgates

| Gate | Bestehen | Reaktion bei Fehlschlag |
| --- | --- | --- |
| Dekodierbarkeit | MP4/MOV mit monotonen Zeitstempeln, mindestens 1920×1080, mindestens 24 fps; Ziel: 3840×2160 bei 30 fps | Datei ablehnen; keine Interpolation oder Upscaling als Ersatz |
| Einheitliche Kamera | Ein einzelnes erkanntes Kamera-/Objektivprofil, konstante Bildgröße und Brennweite | Ablehnen; Clip am Linsen-/Zoomwechsel teilen und getrennt neu aufnehmen |
| Schärfe | Nach szenenadaptiver Selektion bleiben mindestens 70 % der Kandidaten scharf: Varianz des Laplace-Scores mindestens 30 % des 90. Perzentils des Clips; zusätzlich dürfen nicht mehr als 10 % der Schlüsselbilder unter dem absoluten, geräteprofilierten Blur-Grenzwert liegen | Zu schnelle/verwackelte Zeitfenster anzeigen und Neuaufnahme verlangen |
| Belichtung | In höchstens 15 % der Schlüsselbilder sind mehr als 25 % der Pixel gleichzeitig nahezu schwarz oder gesättigt; keine langen Belichtungs-Ausreißer | Warnung bei lokalem Problem, Ablehnung bei einem Raumabschnitt ohne verwertbare Frames |
| Überlapp/Parallaxe | Mindestens 70 % der geprüften zeitlichen Nachbarkanten haben ≥60 % geschätzte Bildüberdeckung **und** mediane Parallaxe 1,5–20° | Langsamer und mit seitlicher Bewegung neu aufnehmen; reine Drehung wird abgelehnt |
| Vollständigkeit | Mindestens 40 verwertbare Keyframes insgesamt und mindestens 12 je räumlich getrenntem Raumcluster; kein Cluster darf weniger als drei Blickrichtungen je exportierter Wand haben | Fehlende Räume/Wände im Vorschaupfad markieren |

Der Laplace-Grenzwert ist absichtlich zweistufig: Ein einziger fixer
Varianz-Wert würde strukturarme, aber scharfe weiße Wände fälschlich ablehnen.
Die finale Entscheidung fällt deshalb stets gemeinsam mit den geometrischen
Gates, nicht durch Schärfe allein.

## Rekonstruktionsvertrag

### Harte Annahmegates

| Bereich | Messung nach globalem Bundle Adjustment/Fusion | Bestehen |
| --- | --- | --- |
| Ein Modell | Alle für den Export beanspruchten Räume liegen in einer verbundenen Kamera-/Match-Komponente; keine getrennten Teilmodelle | Ja, sonst ablehnen |
| Bildregistrierung | ≥95 % aller ausgewählten Keyframes und ≥90 % pro Raumcluster registriert; keine unregistrierte Zeitlücke >2 s innerhalb eines beanspruchten Abschnitts | Ja |
| Geometrische Stützung | Jeder innere Keyframe hat mindestens zwei unabhängig verifizierte Nachbarbeziehungen mit jeweils ≥100 Inliern; jede exportierte Wand/Bodenfläche wird aus ≥3 Kameraposen gestützt | Ja |
| Reprojektion | Median des COLMAP-3D-Punktfehlers ≤1,0 px und 95. Perzentil ≤2,5 px | Ja |
| Schleife/Drift | Mindestens eine erkannt und optimiert geschlossene Schleife; nach Skalierung Endpunktversatz am selben visuellen Ort ≤max(0,10 m, 1 % der Trajektorienlänge) | Ja, wenn eine Rundtour beansprucht wird |
| Anker | Beide Ankerpunkte sind trianguliert und in ≥3 Bildern beobachtet; Leave-one-observation-out-Streuung der Ankerlänge ≤max(0,015 m, 1,5 %) | Ja |
| Fläche | Pro exportierter Wand mindestens 90 % ihrer Länge mit dichter, confidence-gewichteter Unterstützung; 95. Perzentil des Punkt-zu-Ebene-Abstands ≤0,03 m | Ja |
| 2D-Topologie | Geschlossener, nicht selbstschneidender Außenumriss; jede Türöffnung verbindet genau zwei begehbare Bereiche oder ist explizit „unbekannt“; keine erfundene verdeckte Wand | Ja |

Die Inlier-, Reprojektion- und Registrierungswerte sind bewusst messbare
Qualitätsproxies. Sie dürfen nicht als Zentimeter-Genauigkeitsbehauptung
missverstanden werden: Ein konsistentes, aber systematisch falsches Modell kann
kleinen Reprojektionsfehler besitzen. Die Anker- und Schleifen-Gates fangen die
häufigsten globalen Ausreißer zusätzlich ab.

### Maß- und Exportstatus

| Status | Bedingung | Was exportiert werden darf |
| --- | --- | --- |
| `verified` | Alle harten Gates plus unabhängiger Kontrollanker mit Fehler ≤max(0,03 m, 2 %) | GLB und SVG mit Maßen; Prüfbericht und Kontrollfehler beilegen |
| `scale-anchored` | Alle harten Gates, aber kein Kontrollanker | GLB und SVG; deutlich als geschätzte, maßstabskalierte Maße kennzeichnen |
| `review` | Grundgeometrie besteht, aber Kontrollanker liegt bei >2 % bis ≤5 % Fehler oder lokale Abdeckung ist als lückenhaft markiert | Vorschau und Diagnostik, jedoch keine „metrisch verifiziert“-Kennzeichnung |
| `recapture-required` | Ein hartes Gate fällt durch oder Kontrollanker >5 % abweicht | Kein finaler GLB/SVG-Export; Heatmap/Begründung und Aufnahmehinweise ausgeben |

Die 2-%-Grenze des Kontrollankers ist die MVP-Zusage für einen starken,
verifizierten Floorplan. Sie gilt nur für die konkret beobachteten und
exportierten Bereiche. Das Ergebnis ersetzt weder Aufmaß noch einen
rechtsverbindlichen Bauplan.

## Was die Pipeline protokollieren muss

Jeder Lauf schreibt neben GLB/SVG ein maschinenlesbares `quality-report.json`
und einen lesbaren Bericht. Mindestens enthalten sein müssen:

- Quelldatei-Hash, Dauer, Auflösung, FPS, Kamera-Modell und erkannte Wechsel;
- Anzahl Kandidat-/Schlüssel-/registrierter Frames, pro Raumcluster und mit
  Zeitfenstern der Ausschlüsse;
- Schärfe-, Belichtungs-, Überlapp-, Parallax-, Inlier- und
  Reprojektion-Verteilungen (Median, p05, p95);
- Komponenten, Loop-Closure-Kanten und Endpunktversatz;
- Ankerpositionen, eingegebene Länge, Skalierungsfaktor, Ausgleichs- bzw.
  Leave-one-out-Streuung und ggf. Kontrollankerfehler;
- Flächenabdeckung, Ebenenrestfehler, unbekannte/ausgelassene Bereiche und
  finalen Status samt jeder verletzten Regel.

Der SVG-Export muss unbekannte Grenzen gestrichelt und mit `confidence` bzw.
`verified` markieren; er darf Lücken nicht durch eine plausible rechteckige
Raumform stillschweigend schließen. Der GLB-Export muss dieselben
Qualitätsmetadaten im Begleitbericht referenzieren.

## Validierung während der Entwicklung

1. **SLAM-Regression:** Auf Sequenzen mit bekannter Trajektorie ATE und RPE
   nach TUM-Konvention auswerten; ATE testet globale Konsistenz, RPE lokalen
   Drift. Das ist ein Engineering-Regressionstest, nicht der Kunden-Gate.
2. **Wohnungs-Testkorpus:** Mindestens zehn Wohnungen/Raumlayouts mit
   verschiedenen Licht-, Textur- und Türsituationen; für jedes Objekt zwei
   getrennte, mit Lasermaß erhobene Kontrollstrecken sowie ein referenzierter
   2D-Grundriss. Die erste Strecke skaliert, die zweite bleibt blind für die
   Qualitätsauswertung.
3. **Akzeptanzkriterium vor Release:** Kein `verified`-Ergebnis des Testkorpus
   darf den Kontrollanker-Gate verletzen; jeder absichtlich verwackelte,
   unvollständige oder schleifenlose Clip muss mit `recapture-required` enden.
   False-accepts sind schwerer zu gewichten als False-recaptures.
4. **Kalibrierung:** Die Zahlen dieses Dokuments nach den ersten fünf
   repräsentativen Wohnungen anhand von Fehlerverteilungen aktualisieren;
   Änderungen versionieren und im Qualitätsbericht ausgeben.

## Folgen für die Architektur

- Die Videopipeline braucht vor der dichten Depth-Fusion eine explizite
  `capture_audit`-Stufe und muss deren Keyframe-Entscheidungen persistieren.
- SfM/SLAM muss globale Optimierung und Schleifenschluss liefern; unabhängige
  Einzelbild-Posen genügen nicht.
- Der Anker ist ein Erstklasse-Objekt mit Bildbeobachtungen, Unsicherheit und
  separater Rolle (`scale` oder `validation`), nicht bloß ein CLI-Zahlenwert.
- Exporter konsumieren ausschließlich ein Quality-Result mit Status. Sie können
  keinen „finalen“ Floorplan aus einem fehlgeschlagenen Modell erzeugen.
- Für hochstrukturierte, dunkle oder spiegelnde Räume muss die CLI eine
  reproduzierbare Diagnose und Anleitung zur Nachaufnahme ausgeben, statt
  Schwellenwerte still zu lockern.
