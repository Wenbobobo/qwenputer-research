// Top navigation for Qwenputer stage reports
(function() {
  const nav = document.createElement('nav');
  nav.id = 'stage-nav';
  nav.innerHTML = `
    <div class="nav-container">
      <a href="../index.html" class="nav-home">首页</a>
      <div class="nav-stages">
        <a href="stage1.html">Stage 1</a>
        <a href="stage2.html">Stage 2</a>
        <a href="stage3.html">Stage 3</a>
        <a href="stage4.html">Stage 4</a>
        <a href="stage5.html">Stage 5</a>
        <a href="stage6.html">Stage 6</a>
        <a href="stage7.html">Stage 7</a>
        <a href="stage8.html">Stage 8</a>
        <a href="stage9.html">Stage 9</a>
      </div>
    </div>
  `;
  
  // Insert at top of body
  if (document.body) {
    document.body.insertBefore(nav, document.body.firstChild);
    // Add padding to body to account for fixed nav
    document.body.style.paddingTop = '50px';
  }
  
  // Highlight current page
  const currentPage = window.location.pathname.split('/').pop();
  document.querySelectorAll('#stage-nav a').forEach(a => {
    if (a.getAttribute('href') === currentPage || 
        (currentPage === 'index.html' && a.classList.contains('nav-home'))) {
      a.classList.add('active');
    }
  });
})();