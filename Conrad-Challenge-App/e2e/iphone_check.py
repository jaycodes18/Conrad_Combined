"""Run: npx expo export --platform web --output-dir dist-web && python3 -m http.server 8099 -d dist-web
then, in another terminal: python3 e2e/iphone_check.py   (needs `pip install playwright` and `playwright install chromium`)

End-to-end check of the Driven app as an iPhone 15 Pro (Chromium with Playwright's iPhone 15 Pro profile:
393x852 CSS px, 3x density, touch, iOS Safari user agent). Every request in the brief is an assertion here."""
import asyncio, json, sys, time
from playwright.async_api import async_playwright

import os
URL = os.environ.get('DRIVEN_URL', 'http://localhost:8099/')
OUT = os.environ.get('DRIVEN_SHOTS', 'e2e/shots/')
CHROME = os.environ.get('CHROME_PATH')
results = []

def record(name, ok, detail=''):
    results.append({'check': name, 'pass': bool(ok), 'detail': detail})
    print(('PASS ' if ok else 'FAIL ') + name + (' :: ' + detail if detail else ''))

async def open_as(pw, browser, role, name=''):
    device = dict(pw.devices['iPhone 15 Pro'])
    ctx = await browser.new_context(**device, color_scheme='dark')
    if role:
        prof = json.dumps({'role': role, 'driver': name, 'contact': None, 'gps': False, 'voice': True})
        await ctx.add_init_script(f"localStorage.setItem('conrad.profile.v1', {json.dumps(prof)});")
    tiles = []
    ctx.on('request', lambda r: tiles.append(r.url) if 'tile.openstreetmap.org' in r.url else None)
    page = await ctx.new_page()
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    await page.goto(URL)
    await page.wait_for_timeout(2500)
    return page, tiles, errors

async def main():
    async with async_playwright() as pw:
        os.makedirs(OUT, exist_ok=True)
        browser = await pw.chromium.launch(executable_path=CHROME, args=['--no-sandbox']) if CHROME else await pw.chromium.launch()

        # Role picker and logo
        page, _, errors = await open_as(pw, browser, None)
        logo = page.get_by_label('Driven')
        record('Logo from assets/icon.png on the role picker', await logo.count() >= 1 and await logo.first.is_visible())
        src = await page.evaluate("() => [...document.querySelectorAll('img')].map(i => i.src).find(s => s.includes('icon')) || ''")
        record('Logo image is the original app icon file', 'icon' in src, src.rsplit('/', 1)[-1])
        await page.screenshot(path=OUT + '01_role.png')
        record('No runtime errors on launch', not errors, '; '.join(errors)[:200])

        # Teen drive screen
        page, _, errors = await open_as(pw, browser, 'driver', 'Maya')
        await page.screenshot(path=OUT + '02_teen_drive.png')
        record('Header shows the logo', await page.get_by_label('Driven').count() >= 1)
        for label in ['Share', 'Camera', 'Location', 'Voice']:
            record(f'Quick control "{label}" keeps its text label', await page.get_by_text(label, exact=True).count() >= 1)
        icon_fonts = await page.evaluate("() => [...document.querySelectorAll('div,span')].filter(e => getComputedStyle(e).fontFamily.includes('material-community')).length")
        record('Icons render from the MaterialCommunityIcons font', icon_fonts >= 8, f'{icon_fonts} glyphs')
        for tab in ['Drive', 'Trips', 'Family', 'Settings']:
            record(f'Teen tab "{tab}" keeps its text label', await page.get_by_role('tab', name=tab).count() == 1)
        # press animation: scale changes while pressed
        start_btn = page.get_by_label('Start drive')
        box = await start_btn.bounding_box()
        await page.mouse.move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)
        await page.mouse.down(); await page.wait_for_timeout(180)
        pressed = await start_btn.evaluate("e => getComputedStyle(e.firstElementChild || e).transform")
        await page.mouse.move(5, 5); await page.mouse.up()
        record('Buttons spring when pressed', pressed not in ('none', 'matrix(1, 0, 0, 1, 0, 0)'), pressed)

        await page.get_by_text('Practice drills').click(); await page.wait_for_timeout(500)
        await page.screenshot(path=OUT + '03_teen_practice.png')
        await page.get_by_text('Eye-closure drill').click()
        await page.wait_for_timeout(9800)
        banner = page.get_by_text('Open your eyes and look at the road')
        record('Response-check banner appears during eye closure', await banner.count() == 1)
        t1 = await banner.evaluate("e => getComputedStyle(e.parentElement).transform")
        await page.wait_for_timeout(450)
        t2 = await banner.evaluate("e => getComputedStyle(e.parentElement).transform")
        record('Banner pulses (transform changes over time)', t1 != t2, f'{t1} -> {t2}')
        risk = await page.evaluate("() => Math.max(...[...document.querySelectorAll('div')].map(d => d.childNodes.length === 1 && /^\\d{1,3}$/.test(d.textContent) ? +d.textContent : 0))")
        record('Risk number counted up to the critical range', risk >= 90, f'{risk}')
        await page.screenshot(path=OUT + '04_teen_drill.png')
        record('No runtime errors in teen flow', not errors, '; '.join(errors)[:200])

        # Parent map with Leaflet and a card per kid
        page, tiles, errors = await open_as(pw, browser, 'guardian')
        await page.wait_for_timeout(2500)
        frame = page.frame_locator('iframe')
        record('Leaflet map is mounted', await frame.locator('.leaflet-container').count() == 1)
        record('Map requests OpenStreetMap tiles', len(tiles) >= 4, f'{len(tiles)} tile requests, e.g. {tiles[0].split("org/")[-1] if tiles else "none"}')
        cars = frame.locator('.car')
        record('A car marker for each kid', await cars.count() == 2, f'{await cars.count()} markers')
        for kid in ['Maya', 'Arjun']:
            record(f'Card for {kid} on the map', await page.get_by_label(f'Open {kid}').count() == 1)
        record('Cards show a live thumbnail', await page.get_by_text('MODEL VIEW').count() >= 2)
        await page.screenshot(path=OUT + '05_parent_map.png')
        await page.get_by_label('Open Maya').hover()
        await page.mouse.wheel(300, 0); await page.wait_for_timeout(700)
        await page.screenshot(path=OUT + '05b_parent_map_swiped.png')

        await page.frame_locator('iframe').locator('.car .badge').nth(0).click(force=True); await page.wait_for_timeout(1200)
        opened = await page.get_by_text('Risk factors').count() >= 1
        record('Tapping a car on the map opens that kid', opened)
        await page.screenshot(path=OUT + '06_parent_driver.png')
        await page.wait_for_function("((Date.now()/1000)%70) > 53 && ((Date.now()/1000)%70) < 58", timeout=75000)
        await page.wait_for_timeout(600)
        await page.screenshot(path=OUT + '07_parent_driver_episode.png')
        record('Driver screen shows condition, risk and indicators', await page.get_by_text('RISK').count() >= 1 and await page.get_by_text('EYES').count() >= 1)
        await page.get_by_text('Driving history').first.click(); await page.wait_for_timeout(700)
        record('Driving history page opens', await page.get_by_text('THIS WEEK').count() == 1)
        await page.screenshot(path=OUT + '08_parent_history.png')
        await page.get_by_role('tab', name='Alerts').click(); await page.wait_for_timeout(1500)
        record('Alerts tab lists alerts', await page.get_by_text('Eyes off the road').count() >= 1)
        await page.screenshot(path=OUT + '09_parent_alerts.png')
        record('No runtime errors in parent flow', not errors, '; '.join(errors)[:200])
        await browser.close()
    json.dump(results, open(OUT + 'results.json', 'w'), indent=1)
    print(f"{sum(r['pass'] for r in results)}/{len(results)} passed")

asyncio.run(main())
