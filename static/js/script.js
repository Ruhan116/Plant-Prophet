document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('my-crops').addEventListener('click', () => {
        window.location.href = '/mycrops';
    });

    document.getElementById('crop-recommendation').addEventListener('click', () => {
        window.location.href = '/crop_recommendation';
    });

    document.getElementById('articles').addEventListener('click', () => {
        // Assuming you have a route for articles
        window.location.href = '/articles';
    });
});
