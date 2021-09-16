$(document).ready(function() {
    $('a.abstract').click(function() {
        $(this).parent().parent().find(".abstract.hidden").toggleClass('open');
    });
    $('a.summary').click(function() {
        $(this).parent().parent().find(".summary.hidden").toggleClass('open');
        if ($(".slides.hidden").hasClass("open")) {
            $(this).parent().parent().find(".slides.hidden").toggleClass('open');
        }         
    });
    $('a.slides').click(function() {
        $(this).parent().parent().find(".slides.hidden").toggleClass('open');
        if ($(".summary.hidden").hasClass("open")) {
            $(this).parent().parent().find(".summary.hidden").toggleClass('open');
        }
        
    });
    $('a.bibtex').click(function() {
        $(this).parent().parent().find(".bibtex.hidden").toggleClass('open');
    });
    $('.navbar-nav').find('a').removeClass('waves-effect waves-light');
});
