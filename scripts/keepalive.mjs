// Visit the public app so Streamlit Community Cloud's 12h inactivity
// hibernation never triggers; wake it if it already went to sleep.
import puppeteer from "puppeteer-core";

const URL = "https://ronin-rag.streamlit.app";
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH || "/usr/bin/google-chrome",
  args: ["--no-sandbox"],
});
try {
  const page = await browser.newPage();
  await page.goto(URL, { waitUntil: "networkidle2", timeout: 60000 });
  const wake = await page.$$eval("button", (els) =>
    els.findIndex((b) => /get this app back up/i.test(b.innerText))
  );
  if (wake >= 0) {
    const buttons = await page.$$("button");
    await buttons[wake].click();
    console.log("app was asleep — clicked wake, waiting for boot");
    await new Promise((r) => setTimeout(r, 90000));
  }
  await new Promise((r) => setTimeout(r, 20000));
  const title = await page.title();
  console.log("final title:", title);
  if (!/Ronin/i.test(title)) process.exitCode = 1;
} finally {
  await browser.close();
}
