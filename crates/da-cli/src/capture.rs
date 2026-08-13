//! Capture instructions for footage that can support metric floorplans.

use clap::ValueEnum;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RoomType {
    /// Ordinary room with corners and straight walls.
    Rectangular,
    /// Round or curved room.
    Round,
}

pub fn print_capture_guide(room_type: RoomType) {
    println!(
        "AUFNAHMEPLAN – {}",
        match room_type {
            RoomType::Rectangular => "eckiger Raum",
            RoomType::Round => "runder Raum",
        }
    );
    println!("\nVorher");
    println!("  1. Boden und Wandkanten freiräumen; Türen öffnen, Vorhänge hoch.");
    println!("  2. Handy: 1×-Linse, kein Zoom, Hochformat, mindestens 1080p/30 fps; Fokus/Belichtung sperren.");
    println!("  3. Zwei gemessene Referenzen bereitlegen (z. B. 2.000-mm-Zollstock und Türbreite). Jede mindestens dreimal filmen.");
    println!("\nAufnahme");
    println!("  1. Nicht in der Mitte drehen. Langsam mit 0,3–0,8 m/s gehen und das Telefon parallel zur Laufrichtung halten.");
    println!("  2. In etwa 1–1,5 m Abstand an den Wänden entlanggehen; Kamera leicht (ca. 45°) zur Wand und zum Boden neigen.");
    println!("  3. Jede Ecke, Türöffnung und Wandfläche aus mindestens drei Positionen aufnehmen; auf 70–85 % Bildüberlappung achten.");
    match room_type {
        RoomType::Rectangular => {
            println!("  4. An der Tür beginnen und nach einem vollständigen Rundgang exakt dort enden – die letzte Ansicht soll die erste überlappen.");
        }
        RoomType::Round => {
            println!("  4. Zwei vollständige Kreise laufen: außen nahe der Wand, dann innen mit 1–1,5 m Versatz. Anfang und Ende je Kreis überlappen lassen.");
            println!("  5. An vier Stellen langsam vom äußeren zum inneren Kreis wechseln und dabei dieselbe Tür, dasselbe Fenster oder denselben Heizkörper im Bild behalten. Diese Brücken verbinden beide Kreise zu einem Modell.");
            println!("  6. Tür, Fenster, Heizkörper und jede Richtungsänderung zusätzlich frontal und schräg erfassen – sie stabilisieren die Kreisgeometrie.");
        }
    }
    println!("\nVermeiden");
    println!("  - Menschen, Spiegelungen, schnelle Schwenks, Bewegungsunschärfe und dauerhaft verdeckte Wandabschnitte.");
    println!("  - Reines Panorama vom Mittelpunkt: Es liefert kaum Tiefenparallaxe und ist für Grundrisse nicht ausreichend.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn room_types_are_distinct() {
        assert_ne!(RoomType::Rectangular, RoomType::Round);
    }
}
