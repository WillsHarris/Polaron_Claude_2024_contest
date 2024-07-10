document.addEventListener('scroll', () => {
    console.log('Scroll event detected');
    
    const topBar = document.querySelector('.top-bar');
    const logo = document.querySelector('.logo img');
    const main = document.querySelector('main');

    const isScrolled = window.scrollY > 30;

    topBar.style.padding = isScrolled ? '10px 20px' : '20px 20px';
    logo.style.width = isScrolled ? '90px' : '120px';
    main.style.marginTop = isScrolled ? '50px' : '80px';
});

document.querySelector('.reel-container').addEventListener('scroll', function() {
    console.log('Scroll event detected in reel-container');

    const topBar = document.querySelector('.top-bar');
    const logo = document.querySelector('.logo img');
    const main = document.querySelector('main');

    const isScrolled = this.scrollTop > 400;

    topBar.style.padding = isScrolled ? '10px 20px' : '20px 20px';
    logo.style.width = isScrolled ? '90px' : '120px';
    main.style.marginTop = isScrolled ? '50px' : '80px';
});

document.querySelector('.reel-container').addEventListener('scroll', function() {
    console.log('Scroll event detected in reel-container');

    const topBar = document.querySelector('.top-bar');
    const logo = document.querySelector('.logo img');
    const main = document.querySelector('main');

    const isScrolled = this.scrollTop > 400;

    topBar.style.padding = isScrolled ? '10px 20px' : '20px 20px';
    logo.style.width = isScrolled ? '90px' : '120px';
    main.style.marginTop = isScrolled ? '50px' : '80px';
});