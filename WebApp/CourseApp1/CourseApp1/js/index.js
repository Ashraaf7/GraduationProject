//menu
let menuButton=document.getElementById("nav_button");
let closeButton=document.getElementById("nav_button_close");
let menuul=document.getElementById("menu_ul");
menuButton.addEventListener("click",function(){
    closeButton.style.display='block';
        menuul.style.right='0';
})
//close menu
closeButton.addEventListener("click",function(){
    closeButton.style.display='none';
        menuul.style.right='-220vw';
})

        let time_height=document.getElementsByClassName("time");
        let line_height=document.getElementById("line");
        line_height.style.height=time_height[0].clientHeight+"px";
    
        let even_container=document.querySelectorAll('.roadmap_container:nth-child(even)');
        let odd_container=document.querySelectorAll('.roadmap_container:nth-child(odd)');
        let even_container_span=document.querySelectorAll('.roadmap_container:nth-child(even) span');
        let odd_container_span=document.querySelectorAll('.roadmap_container:nth-child(odd) span');
        even_container.forEach(element => {
            element.classList.add('start-50');
        });
        odd_container.forEach(element => {
            element.classList.add('start-0');
        });
        even_container_span.forEach(element => {
            element.classList.add('start-0');
        });
        odd_container_span.forEach(element => {
            element.classList.add('start-100');
        });
