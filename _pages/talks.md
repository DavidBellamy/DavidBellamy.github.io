---
layout: page
permalink: /talks/
title: talks
description: 
display_categories: [journal clubs, research talks] # add invited talks hopefully some day
nav: true
---

<div class="talks">

{% for category in page.display_categories %}
  <h2 class="category">{{category}}</h2>
    {% assign categorized_talks = site.talks | where: "category", category %}
    <!-- sort by presentation date -->
    <!-- <div class="container"> -->
        <!-- <div class="row row-cols-2"> -->
            {% for talk in categorized_talks %}
                {% include talks.html %}
            {% endfor %}
        <!-- </div> -->
    <!-- </div> -->
{% endfor %}

</div>
