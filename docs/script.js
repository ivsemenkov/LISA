const header = document.querySelector('.site-header');
const menuButton = document.querySelector('.nav-toggle');
const navLinks = document.querySelectorAll('.site-nav a');

function closeMenu() {
  header?.removeAttribute('data-open');
  menuButton?.setAttribute('aria-expanded', 'false');
}

menuButton?.addEventListener('click', () => {
  const isOpen = header.getAttribute('data-open') === 'true';
  if (isOpen) {
    closeMenu();
    return;
  }
  header.setAttribute('data-open', 'true');
  menuButton.setAttribute('aria-expanded', 'true');
});

navLinks.forEach((link) => link.addEventListener('click', closeMenu));

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeMenu();
});

const copyCitationButton = document.querySelector('[data-copy-citation]');
const citationCode = document.querySelector('#bibtex-citation code');
const copyCitationLabel = copyCitationButton?.querySelector('span');
let copyResetTimer;

copyCitationButton?.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(citationCode.textContent.trim());
    copyCitationLabel.textContent = 'Copied!';
  } catch {
    copyCitationLabel.textContent = 'Copy failed';
  }

  clearTimeout(copyResetTimer);
  copyResetTimer = window.setTimeout(() => {
    copyCitationLabel.textContent = 'Copy BibTeX';
  }, 1800);
});
