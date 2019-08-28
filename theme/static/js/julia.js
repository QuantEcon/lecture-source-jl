/* julia.js v1.0 */

$(function () {

	/* Popup for choosing PDF download */
	$('a[rel*=leanModal]').leanModal({ top : 200, closeButton: ".modal_close" });


	$('.pdf-options .py a').on('click', function() {
		$('.modal_close').click();
		ga('send', 'event', 'download', 'pdf', 'QuantEconlectures-python3.pdf');
	});

	$('.pdf-options .jl a').on('click', function() {
		$('.modal_close').click();
		ga('send', 'event', 'download', 'pdf', 'QuantEconlectures-julia.pdf');
	});

	/* Lectures How To */
	$('.how-to .toggle').click(function(e){
		$('.how-to-content').slideToggle();
		e.preventDefault();
	});

	/* Collapsed code block */
	$("div[class^='collapse']").each(function(){
		$('.highlight', this).after('<a href="#" class="toggle toggle-less" style="display:none;"><span class="icon icon-angle-double-up"></span><em>Show less...</em></a>');
		$('.highlight', this).after('<a href="#" class="toggle toggle-more"><span class="icon icon-angle-double-down"></span><em>Show more...</em></a>');
	});

	$('div[class^="collapse"]').on('click', '.toggle', function(e){
		var codeBlock = $(this).parents('div[class^="collapse"]');
    	if ( codeBlock.hasClass('expanded') ) {
    		codeBlock.removeClass('expanded').find('.toggle').toggle();
    		$('html, body').animate({
        		scrollTop: $(codeBlock).offset().top - 50
    		}, 400);
    	} else {
    		codeBlock.addClass('expanded').find('.toggle').toggle();
    	}
    	e.preventDefault();
	});

	/* Read news feed and render on homepage */
	if ( $('.news-feed').length > 0 ) {
		$.ajax({
			crossDomain: true,
			url: "https://quantecon.org/news/json/feed.json",
			dataType: "jsonp",
			success: function(data) {
	             var item = '';
	             var html = '<h2>QuantEcon news</h2><ul>';
	             $.each(data.items, function (i, e) {
	             	item = '<li><a href="' + e.url +'">' + e.title + '</a> <p class="date">' + e.date + '</p><p class="summary">' + e.summary + '</p></li>';
	             	html = html + item;
	             });
	             html = html + '</ul><p class="more"><a href="http://quantecon.org/news">More news</a></p>';
	             $('.news-feed .container').append(html);
			}
		});
	}

	/* path prefix for individual lectures */
	var index = window.location.pathname.indexOf('/', window.location.pathname.indexOf('/') + 1);
	var lang = window.location.pathname.substring(0, index + 1);
	lang ? lang : lang = '/'

	/* Display a notebook execution badge on lecture pages */
	if ( $('#executability_status_badge').length > 0 ) {
		load_this_page_badge(lang);
	}

	/* Display a notebook execution status table */
	if ( $('#status_table').length > 0 ) {
		load_status_table(lang);
	}

	/* Display coverage badges */
	load_percentages(lang);


});

// Notebook Launcher popup
if ( document.getElementById('launchButton') ) {
	const template = document.getElementById('launchModal');
	template.style.display = 'block';
	tippy('#launchButton', {
	  content: template,
	  theme: 'light-border',
	  animation: 'shift-away',
	  inertia: true,
	  duration: [200,200],
	  arrow: true,
	  arrowType: 'round',
	  delay: [200, 200],
	  interactive: true
	});
}

// Clone popup
if ( document.getElementById('cloneButton') ) {
	const template = document.getElementById('cloneModal');
	template.style.display = 'block';
	tippy('#cloneButton', {
	  content: template,
	  theme: 'light-border',
	  animation: 'shift-away',
	  inertia: true,
	  duration: [200,200],
	  arrow: true,
	  arrowType: 'round',
	  delay: [200, 200],
	  interactive: true
	});
}

// Copies a string to the clipboard
function copyToClipboard(text) {
    if (window.clipboardData && window.clipboardData.setData) {
        // IE specific code path to prevent textarea being shown while dialog is visible.
        return clipboardData.setData("Text", text);

    } else if (document.queryCommandSupported && document.queryCommandSupported("copy")) {
        var textarea = document.createElement("textarea");
        textarea.textContent = text;
        textarea.style.position = "fixed";  // Prevent scrolling to bottom of page in MS Edge.
        document.body.appendChild(textarea);
        textarea.select();
        try {
            return document.execCommand("copy");  // Security exception may be thrown by some browsers.
        } catch (ex) {
            console.warn("Copy to clipboard failed.", ex);
            return false;
        } finally {
            document.body.removeChild(textarea);
        }
    }
}
