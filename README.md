# RL Sphero upravljanje

UPOZORENJE: trenutno ovo rješenje ne radi najbolje

Ovaj repozitorij nadovezuje se na: https://github.com/mkrizmancic/sphero_formation/tree/0ac14aad3dd1a0af26f191c017e213279eebd52e
Koristi se _openai gym_ toolkit za RL (dobiven modifikacijom već postojećeg _openai\_ros_ paketa - http://wiki.ros.org/openai_ros).

## Ovisnosti
Potrebno je instalirati _openai gym_ paket:
```bash
pip install gym
```
Također, _FFmpef_ kako bi se graf mogao spremiti kao video/gif:
```bash
sudo apt install ffmpeg
``` 
## O paketu
Ovaj repozitorij dodatno sadrži nekoliko skripti koje povezuju RL sa ROS-om i Stage simulatorom
1. _stage\_connection.py_ - klasa koja sadrži _pause, unpause, reset_ funkcije povezane sa Stage simulatorom (relikvija _openai\_ros_ paketa, moguće da će biti izbačena u budućnosti)
1. _robot\_stage\_env.py_ - najosnovnija verzija _gym environment-a_.
1. _sphero\_env.py_ - sve što povezuje Sphero robota i ROS (_subscribers_ i _publishers_ )
1. _sphero\_world.py_ - sve što povezuje Sphero robota i RL (definiranje akcija, stanja, nagrada kao i njihovo _handle-anje_)
1. _qlearn.py_ - klasa koja sadrži Q-learn jednadžbe
1. _start\_qlearning.py_ - skripta koja obavlja posao "učenja". Za zadani broj epizoda i koraka u epizodi proračunava i ažurira Q vrijednosti
1. _qlearn\_params.yaml_ - sadrži parametre vezane za učenje

## Pokretanje
Prije pokretanja, potrebno je pozicionirati se unutar paketa:
```bash
cd catkin_ws/src/sphero_formation
```
te preuzeti _stage\_ros_ paket naredbom:
```bash
git submodule update --init
```
Zatim je potrebno kopirati paket u _\src_:
```bash
cp -r ~/catkin_ws/src/sphero_formation/stage_ros ~/catkin_ws/src
```
Ostalo je isto kao i _sphero\_formation_ paket (u _reynolds\_sim.launch_ je dodano pokretanje RL _node-a_)
