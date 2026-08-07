from poke_env import ServerConfiguration, LocalhostServerConfiguration, AccountConfiguration
from poke_env.player import RandomPlayer
import asyncio
import logging

logging.basicConfig(level=logging.WARNING)

class MyPlayer(RandomPlayer):
    def choose_move(self, battle):
        dump_battle_state(battle)
        return self.choose_random_move(battle)

async def main():
    print("Hello from bot!")

    player = MyPlayer(
        account_configuration=AccountConfiguration(
            username="SUPER OMEGA BOT",
            password=None,
        ),
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        log_level=logging.WARNING,
    )

    await player.ladder(1)

    print(player.rating)

def dump_battle_state(battle):
    print(f"=== Turn {battle.turn} ===")

    print("\n--- Your team ---")
    for mon in battle.team.values():
        print(f"{mon.species}: {mon.current_hp_fraction*100:.0f}% HP, "
              f"status={mon.status}, boosts={mon.boosts}, "
              f"moves={[m.id for m in mon.moves.values()]},"
              f"item={mon.item}, ability={mon.ability},"
              f"EVs: {mon.evs}")

    print("\n--- Opponent's known team ---")
    for mon in battle.opponent_team.values():
        print(f"{mon.species}: {mon.current_hp_fraction*100:.0f}% HP, "
              f"status={mon.status}, revealed_moves={[m.id for m in mon.moves.values()]}, "
              f"item={mon.item}, ability={mon.ability},"
              f"EVs: {mon.evs}, IVs: {mon.ivs},")

    print("\n--- Active Pokémon ---")
    print(f"You: {battle.active_pokemon.species}")
    print(f"Opponent: {battle.opponent_active_pokemon.species}")

    print("\n--- Available actions ---")
    print(f"Moves: {[m.id for m in battle.available_moves]}")
    print(f"Switches: {[s.species for s in battle.available_switches]}")

    print("\n--- Field state ---")
    print(f"Weather: {battle.weather}")
    print(f"Fields (terrain/trick room/etc): {battle.fields}")
    print(f"Your side conditions: {battle.side_conditions}")
    print(f"Opponent side conditions: {battle.opponent_side_conditions}")
    
async def two_player_battle():
    player1 = MyPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
    )
    player2 = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
    )
    await player1.battle_against(player2, n_battles=1)
    
    print(player1.n_won_battles / player1.n_finished_battles)
    print(player2.n_won_battles / player2.n_finished_battles)



if __name__ == "__main__":
    asyncio.run(main())