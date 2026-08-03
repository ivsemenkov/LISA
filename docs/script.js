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
