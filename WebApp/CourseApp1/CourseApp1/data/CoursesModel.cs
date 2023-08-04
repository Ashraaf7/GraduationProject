using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace CourseApp1.data
{
    [MetadataType(typeof(CourseMetadata))]
    public partial class Cours
    {

        //adding property
        public HttpPostedFileBase ImageFile { get; set; }
    }
    public class CourseMetadata
    {
        public int ID { get; set; }
        [Display(Name ="Course Name")]
        public string Name { get; set; }
        public Nullable<System.DateTime> Creation_Date { get; set; }
        public string Description { get; set; }
        public int Category_id { get; set; }
        public Nullable<int> Trainer_id { get; set; }
        [Url]
        public string CourseLink { get; set; }
        public Nullable<bool> IsYoutube { get; set; }
        [Display(Name = "Course Image")]
        public string ImgPath { get; set; }
    }
}