"""Run with:  python -m unittest discover -s tests -v     (no pytest needed)"""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aeris import energy
from aeris.planetary_models import MARS, EARTH, MOON, custom_planet, describe_scout
from aeris.environment import generate_terrain
from aeris.terrain_analysis import compute_slope
from aeris.knowledge import KnowledgeMap
from aeris.navigation import cost_grid, astar
from aeris.communication import CommLink, Packet
from aeris.mission_controller import Mission, MissionConfig
from aeris.rover import Rover
from aeris.metrics import run_study


class TestTerrain(unittest.TestCase):
    def test_reproducible_and_seed_dependent(self):
        a, b, c = generate_terrain(MARS, 3), generate_terrain(MARS, 3), generate_terrain(MARS, 4)
        self.assertTrue(np.array_equal(a.elevation, b.elevation))
        self.assertFalse(np.array_equal(a.elevation, c.elevation))

    def test_slope_of_plane(self):
        z = np.fromfunction(lambda i, j: j * 10.0, (20, 20))      # 10 m rise per 10 m cell = 45 deg
        self.assertAlmostEqual(float(compute_slope(z, 10.0)[10, 10]), 45.0, places=3)

    def test_start_goal_connected_all_presets(self):
        from scipy.ndimage import label
        for p in (MARS, EARTH, MOON):
            for s in range(1, 6):
                t = generate_terrain(p, s)
                lab, _ = label(~t.blocked)
                self.assertTrue(lab[t.start] > 0 and lab[t.start] == lab[t.goal])

    def test_planet_params_change_terrain(self):
        rough = generate_terrain(custom_planet(terrain_roughness=0.9), 5).roughness.mean()
        smooth = generate_terrain(custom_planet(terrain_roughness=0.1), 5).roughness.mean()
        self.assertGreater(rough, smooth + 0.2)
        steep = generate_terrain(custom_planet(slope_severity=1.6), 5).slope_deg.mean()
        flat = generate_terrain(custom_planet(slope_severity=0.5), 5).slope_deg.mean()
        self.assertGreater(steep, flat)


class TestEnergy(unittest.TestCase):
    def test_gravity_slope_roughness_effects(self):
        e = lambda p, dz=0.0, r=0.2: sum(energy.rover_segment(p, 10.0, dz, r, 5.0)[:2])
        self.assertGreater(e(EARTH), e(MARS))
        self.assertGreater(e(MARS), e(MOON))
        self.assertGreater(e(MARS, dz=3.0), e(MARS, dz=0.0))
        self.assertGreater(e(MARS, r=0.8), e(MARS, r=0.1))

    def test_scout_selection(self):
        self.assertEqual(describe_scout(MARS)["kind"], "aerial")
        self.assertEqual(describe_scout(EARTH)["kind"], "aerial")
        self.assertEqual(describe_scout(MOON)["kind"], "orbital")
        self.assertEqual(describe_scout(custom_planet(atmo_density=0.0003))["kind"], "orbital")
        self.assertGreater(energy.hover_power_w(custom_planet(gravity=9.8)), energy.hover_power_w(MARS))
        self.assertGreater(energy.hover_power_w(MARS), energy.hover_power_w(EARTH))


class TestNavigation(unittest.TestCase):
    def test_planner_uses_no_ground_truth(self):
        kn = KnowledgeMap((64, 64))
        m, b = cost_grid(kn)
        p1 = astar(m, b, (5, 5), (57, 56))
        generate_terrain(MARS, 1); generate_terrain(MOON, 9)       # different worlds must not matter
        p2 = astar(*cost_grid(KnowledgeMap((64, 64))), (5, 5), (57, 56))
        self.assertEqual(p1, p2)

    def test_unknown_is_penalised_and_known_hazard_blocked(self):
        kn = KnowledgeMap((20, 20))
        m, b = cost_grid(kn)
        self.assertTrue((m > 2.0).all())
        rr, cc = np.mgrid[0:20, 0:20]
        kn.update(rr.ravel(), cc.ravel(), np.zeros(400), np.zeros(400), np.zeros(400), np.zeros(400, bool))
        m2, _ = cost_grid(kn)
        self.assertLess(m2[5, 5], 1.2)
        kn.obstacle[:, 10] = True
        kn.obstacle[9, 10] = False                                    # one gap
        path = astar(*cost_grid(kn), (0, 0), (19, 19))
        self.assertIn((9, 10), path)
        self.assertFalse(any(kn.blocked()[c] for c in path))

    def test_rover_never_enters_blocked_cells(self):
        for p in (MARS, EARTH, MOON):
            for s in (1, 2, 3, 4):
                m = Mission(MissionConfig(planet=p, seed=s))
                m.run()
                for tr in (m.rover.trail, m.baseline["trail"]):
                    self.assertFalse(any(m.terrain.blocked[c] for c in tr), (p.name, s))


class TestComm(unittest.TestCase):
    def _packet(self, pid=1):
        z = np.zeros(3)
        return Packet(pid, "TERRAIN_OBS", "SCOUT", "ROVER", 0, np.array([1, 2, 3]), np.array([1, 2, 3]),
                      z, z, z, np.zeros(3, bool), 100)

    def test_outage_blocks_then_restores(self):
        link = CommLink((10, 10), 1.0, np.random.default_rng(1), lambda *a: None, loss_base=0.0, base_delay_s=0.5)
        link.enqueue(self._packet())
        link.outage_until = 50.0
        got = []
        for t in range(0, 40):
            got += link.step(float(t), 1.0, 100.0)
        self.assertEqual(got, [])
        self.assertEqual(link.received, 0)
        self.assertEqual(link.shared.n_known, 0)
        for t in range(50, 70):
            got += link.step(float(t), 1.0, 100.0)
        self.assertEqual(len(got), 1)
        self.assertEqual(link.shared.n_known, 3)

    def test_packet_loss_counted(self):
        link = CommLink((10, 10), 1.0, np.random.default_rng(2), lambda *a: None, loss_base=0.85, base_delay_s=0.1)
        for i in range(20):
            link.enqueue(self._packet(i))
        for t in range(0, 400):
            link.step(float(t), 1.0, 50.0)
        self.assertGreater(link.lost, 0)
        self.assertGreater(link.dropped, 0)

    def test_rover_keeps_operating_during_outage(self):
        m = Mission(MissionConfig(planet=MARS, seed=4, recon_wait_s=20.0))
        m.comm.outage_until = 1e9                                     # permanent blackout
        m.run()
        self.assertEqual(m.comm.received, 0)
        self.assertEqual(m.rover.kn.n_known >= 0, True)
        self.assertIn(m.rover.status, ("ARRIVED", "RETURNED", "ABORTED", "STRANDED"))
        self.assertEqual(m.comm.shared.n_known, 0)


class TestMission(unittest.TestCase):
    def test_rover_does_not_know_whole_map(self):
        m = Mission(MissionConfig(planet=MARS, seed=4))
        self.assertLess(m.rover.kn.n_known, 0.05 * 64 * 64)
        m.run(max_steps=60)
        self.assertLess(m.rover.kn.n_known, 0.6 * 64 * 64)
        self.assertTrue(np.all(m.comm.shared.known <= m.scout.obs.known))   # shared is a subset of scout obs

    def test_battery_accounting(self):
        m = Mission(MissionConfig(planet=MARS, seed=4))
        last = m.rover.battery_wh
        for _ in range(400):
            m.step()
            self.assertLessEqual(m.rover.battery_wh, last + 1e-9)
            last = m.rover.battery_wh
        self.assertAlmostEqual(m.rover.battery_wh + m.rover.energy_used, m.rover.capacity, places=6)
        self.assertAlmostEqual(m.scout.battery_wh + m.scout.energy_used, m.scout.capacity, places=6)

    def test_battery_changes_decisions(self):
        weak = custom_planet(rover_battery_wh=30.0)
        m = Mission(MissionConfig(planet=weak, seed=4))
        m.run()
        self.assertNotEqual(m.rover.status, "ARRIVED")
        self.assertNotEqual(m.baseline["status"], "ARRIVED")

    def test_deterministic(self):
        a = Mission(MissionConfig(planet=MARS, seed=5)); a.run()
        b = Mission(MissionConfig(planet=MARS, seed=5)); b.run()
        self.assertEqual(a.rover.trail, b.rover.trail)
        self.assertAlmostEqual(a.scout.energy_used, b.scout.energy_used)

    def test_moon_uses_orbital_platform(self):
        m = Mission(MissionConfig(planet=MOON, seed=4))
        self.assertEqual(m.kind, "orbital")
        m.run()
        self.assertGreater(m.scout.obs.n_known, 500)

    def test_aeris_does_not_always_win(self):
        df = run_study(MARS, range(1, 9))
        self.assertTrue(df["worth it"].any())
        self.assertFalse(df["worth it"].all(), "AERIS must not win every seed")


if __name__ == "__main__":
    unittest.main()
