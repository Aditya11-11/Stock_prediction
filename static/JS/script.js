function sumit() {
            
    // var imageUrl1 = "{{ url_for('static' , filename = 'close_graph.png')}}";
            
    // Define the image source URL
    var imageUrl1 = "/static/GRAPH_IMG/close_graph.png";
    var imgElement1 = document.getElementById("myImage1");
    imgElement1.src = imageUrl1;

    var imageUrl2 = "/static/GRAPH_IMG/ma100_graph.png";
    var imgElement2 = document.getElementById("myImage2");
    imgElement2.src = imageUrl2;
    
    var imageUrl3 = "/static/GRAPH_IMG/ma200_graph.png";
    var imgElement3 = document.getElementById("myImage3");
    imgElement3.src = imageUrl3;
    
    var imageUrl4 = "/static/GRAPH_IMG/prediction_graph.png";
    var imgElement4 = document.getElementById("myImage4");
    imgElement4.src = imageUrl4;
    
    var imageUrl5 = "/static/GRAPH_IMG/prediction_orignal_graph.png";
    var imgElement5 = document.getElementById("myImage5");
    imgElement5.src = imageUrl5;
}
document.addEventListener("DOMContentLoaded",function(){

    if (document.getElementById("identifier")) {
        if (document.getElementById("identifier").innerHTML === 'True') {
            sumit();
          
        }
        
    }
});